# dia/model.py

import os
import logging
import time
import dac  # Keep this import name
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # <<< ADDED Import for safetensors

from .audio import audio_to_codebook, codebook_to_audio
from .config import (
    DiaConfig,
)  # Assuming this is the Pydantic config for model structure
from .layers import DiaModel, KVCache  # Assuming these are the nn.Module definitions

# --- Get a logger instance for this module ---
logger = logging.getLogger(__name__)

# Optional: Add a check after import to verify the library looks correct
# Note: We now expect 'utils' based on original code
if (
    not hasattr(dac, "utils")
    or not hasattr(dac.utils, "download")
    or not hasattr(dac, "DAC")
):
    logger.warning(
        "The imported 'dac' module does not appear to have the 'utils.download' structure expected by the original Dia code."
    )
    logger.warning(
        "Ensure 'descript-audio-codec' is installed correctly (pip install descript-audio-codec)."
    )
    # If this check fails, _load_dac_model will likely raise an error later anyway.


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    use_cfg_filter: bool,
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    """Samples the next token based on logits, temperature, and top_p."""
    if temperature == 0.0:
        # Greedy sampling
        return torch.argmax(logits_BCxV, dim=-1)

    # Apply temperature scaling
    logits_BCxV = logits_BCxV / temperature

    # Apply CFG Top-K filtering (optional)
    if use_cfg_filter and cfg_filter_top_k is not None:
        # Get top K values and indices
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
        # Create a mask to keep only top K logits
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask.scatter_(
            dim=-1, index=top_k_indices_BCxV, value=False
        )  # Set top K positions to False (don't mask)
        # Mask out logits not in the top K
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    # Apply Top-P (Nucleus) sampling
    if top_p < 1.0:
        # Convert logits to probabilities
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        # Sort probabilities in descending order
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(
            probs_BCxV, dim=-1, descending=True
        )
        # Calculate cumulative probabilities
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        # Create mask for tokens to remove (those exceeding top_p threshold)
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        # Shift the mask: keep the first token that crosses the threshold
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[
            ..., :-1
        ].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0  # Always keep the most probable token

        # Scatter the mask back to the original order
        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        # Apply the mask to the logits
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    # Calculate final probabilities after filtering
    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    # Sample from the filtered distribution
    # multinomial expects probabilities for each item in the batch
    sampled_indices_BC = torch.multinomial(
        final_probs_BCxV, num_samples=1
    )  # Shape [B*C, 1]
    sampled_indices_C = sampled_indices_BC.squeeze(
        -1
    )  # Shape [B*C] -> should be [C] if input was [C,V]
    return sampled_indices_C


class Dia:
    """
    Main class for the Dia Text-to-Speech model, handling loading and generation.
    """

    def __init__(self, config: DiaConfig, device: torch.device = torch.device("cuda")):
        """
        Initializes the Dia model structure based on the provided configuration.
        Does not load weights here.

        Args:
            config: The DiaConfig object defining model parameters.
            device: The torch device (e.g., 'cuda', 'cpu') the model should eventually run on.
                    Note: The model is instantiated but not moved to the device here.
        """
        super().__init__()
        logger.info(
            f"Initializing Dia model structure with config version: {config.version}"
        )
        self.config = config
        # Store the target device, but don't move the model yet. Loading weights will handle device placement.
        self.target_device = device
        # Instantiate the underlying PyTorch model based on the config
        self.model = DiaModel(config)
        self.dac_model = None  # DAC model will be loaded separately
        logger.info("Dia model structure initialized.")

    @classmethod
    def load_model_from_files(
        cls,
        config_path: str,
        weights_path: str,
        device: torch.device = torch.device("cuda"),
    ) -> "Dia":
        """
        Loads the Dia model from local configuration and weights files.
        Handles both .pth and .safetensors weight formats.

        Args:
            config_path: Path to the configuration JSON file (e.g., 'config.json').
            weights_path: Path to the model weights file (e.g., 'model.pth' or 'model.safetensors').
            device: The torch device ('cuda', 'cpu', etc.) to load the model onto.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or weights file is not found.
            ValueError: If the weights file format is unsupported.
            RuntimeError: If there is an error loading the config, weights, or DAC model.
        """
        logger.info(f"Loading Dia model from local files:")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Weights: {weights_path}")
        logger.info(f"  Target Device: {device}")

        # 1. Load Configuration
        try:
            config = DiaConfig.load(config_path)
            if config is None:
                # DiaConfig.load returns None on FileNotFoundError
                logger.error(f"Configuration file not found at {config_path}")
                raise FileNotFoundError(
                    f"Configuration file not found at {config_path}"
                )
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(
                f"Error loading or validating configuration from {config_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to load configuration from {config_path}"
            ) from e

        # 2. Instantiate Model Structure
        # Pass the target device during instantiation if the underlying DiaModel supports it,
        # otherwise, we move it later. Assuming __init__ doesn't take device for now.
        dia_instance = cls(
            config, device
        )  # Pass device mainly for storing target_device

        # 3. Load Weights (State Dictionary)
        try:
            logger.info(f"Loading weights from: {weights_path}")
            weights_filename = os.path.basename(weights_path)
            state_dict = None

            if weights_filename.endswith(".safetensors"):
                logger.info(
                    "Detected .safetensors format. Loading using safetensors library."
                )
                # load_file loads directly to the specified device
                state_dict = load_file(weights_path, device=str(device))
                logger.info("Safetensors weights loaded.")
            elif weights_filename.endswith(".pth"):
                logger.info("Detected .pth format. Loading using torch.load.")
                # torch.load needs map_location to load onto the correct device
                state_dict = torch.load(weights_path, map_location=device)
                logger.info("PyTorch weights (.pth) loaded.")
            else:
                logger.error(
                    f"Unsupported weights file format: {weights_filename}. Expected .pth or .safetensors."
                )
                raise ValueError(f"Unsupported weights file format: {weights_filename}")

            # Load the state dictionary into the model structure
            logger.info("Applying loaded weights to the model structure...")
            # Use strict=True by default to catch mismatches. Can be set to False if needed for specific conversions (e.g., BF16 -> FP32 partial loads)
            dia_instance.model.load_state_dict(state_dict, strict=True)
            logger.info("Weights applied successfully.")

        except FileNotFoundError:
            logger.error(f"Weights file not found at {weights_path}")
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        except Exception as e:
            logger.error(
                f"Error loading weights from {weights_path}: {e}", exc_info=True
            )
            raise RuntimeError(f"Error loading weights from {weights_path}") from e

        # 4. Move Model to Device and Set Eval Mode
        logger.info(f"Moving model to device: {device}...")
        dia_instance.model.to(device)
        logger.info("Setting model to evaluation mode...")
        dia_instance.model.eval()

        # 5. Load Associated DAC Model
        logger.info("Loading associated DAC model...")
        dia_instance._load_dac_model()  # This will log its own progress/errors

        logger.info("Dia model fully loaded and ready.")
        return dia_instance

    # REMOVED from_pretrained - Responsibility moved to engine.py
    # @classmethod
    # def from_pretrained(...) -> "Dia": ...

    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model using the original project's method."""
        if self.dac_model is not None:
            logger.info("DAC model already loaded.")
            return

        # Verify the imported module has the necessary structure expected by original code
        if (
            not hasattr(dac, "utils")
            or not hasattr(dac.utils, "download")
            or not hasattr(dac, "DAC")
        ):
            logger.error(
                "Imported 'dac' module structure mismatch. Expected 'dac.utils.download()' and 'dac.DAC'."
            )
            logger.error(
                "Ensure 'descript-audio-codec' is installed correctly via pip."
            )
            raise RuntimeError(
                "Failed to load DAC model: required functions/structure missing from 'dac' module."
            )

        try:
            # Use the original method found in the Dia repository
            logger.info("Downloading/finding DAC model using dac.utils.download()...")
            # This assumes dac.utils.download() handles caching internally
            dac_model_path = dac.utils.download()
            logger.info(f"DAC model path determined: {dac_model_path}")

            logger.info("Loading DAC model from path...")
            # Load DAC model and move it to the same device as the main Dia model
            dac_model = dac.DAC.load(dac_model_path).to(self.target_device)
            logger.info("DAC model loaded successfully.")

        except AttributeError as ae:
            logger.error(
                f"AttributeError loading DAC model: '{ae}'. The installed 'descript-audio-codec' version might be incompatible with Dia's original code which expects 'dac.utils.download()'."
            )
            logger.error(
                "Please check for specific version requirements of 'descript-audio-codec' for Dia, or potential installation issues."
            )
            raise RuntimeError(
                "Failed to load DAC model due to incompatible library version or structure"
            ) from ae
        except Exception as e:
            logger.error(f"General error loading DAC model: {e}", exc_info=True)
            raise RuntimeError("Failed to load DAC model") from e

        self.dac_model = dac_model

    def _create_attn_mask(
        self,
        q_padding_mask_1d: torch.Tensor,
        k_padding_mask_1d: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Creates the attention mask (self or cross) based on padding masks.
        Mimics JAX segment ID logic where attention is allowed between non-padding tokens
        OR between padding tokens, but not across the boundary.

        Args:
            q_padding_mask_1d: Boolean tensor [Batch, SeqLenQ] where True indicates non-padding.
            k_padding_mask_1d: Boolean tensor [Batch, SeqLenK] where True indicates non-padding.
            is_causal: If True, applies an additional causal mask (for decoder self-attention).

        Returns:
            Boolean attention mask tensor [Batch, 1, SeqLenQ, SeqLenK] ready for F.scaled_dot_product_attention.
        """
        B1, Tq = q_padding_mask_1d.shape
        B2, Tk = k_padding_mask_1d.shape
        if B1 != B2:
            logger.warning(
                f"Query ({B1}) and key ({B2}) batch dimensions do not match in _create_attn_mask"
            )
        assert B1 == B2, "Query and key batch dimensions must match"

        # Expand masks for broadcasting: [B, Tq, 1] and [B, 1, Tk]
        p_mask_q = q_padding_mask_1d.unsqueeze(2)
        p_mask_k = k_padding_mask_1d.unsqueeze(1)

        # True where a non-padding query token attends to a non-padding key token
        non_pad_attends_non_pad = p_mask_q & p_mask_k  # Shape [B, Tq, Tk]
        # True where a padding query token attends to a padding key token
        pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

        # Combine: Attention is allowed if tokens are both non-padding OR both padding.
        mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

        if is_causal:
            # Apply causal mask for self-attention (query cannot attend to future keys)
            if Tq != Tk:
                logger.warning(f"Causal mask requested but Tq ({Tq}) != Tk ({Tk})")
            assert (
                Tq == Tk
            ), "Causal mask requires query and key sequence lengths to be equal"
            # Create lower triangular matrix (True allows attention)
            causal_mask_2d = torch.tril(
                torch.ones((Tq, Tk), dtype=torch.bool, device=self.target_device)
            )
            # Combine with padding compatibility mask
            mask = mask & causal_mask_2d  # Shape [B, Tq, Tk]

        # Add head dimension for broadcasting: [B, 1, Tq, Tk]
        return mask.unsqueeze(1)

    def _prepare_text_input(
        self, text: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes text prompt into byte tokens, pads to max length,
        and creates position IDs and padding mask.

        Args:
            text: The input text string.

        Returns:
            Tuple containing:
                - src_tokens: Padded token IDs [1, SeqLen].
                - src_positions: Position IDs [1, SeqLen].
                - src_padding_mask: Boolean mask (True=non-pad) [1, SeqLen].
                - enc_self_attn_mask: Attention mask for encoder [1, 1, SeqLen, SeqLen].
        """
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length
        logger.debug(
            f"Preparing text input. Max length: {max_len}, Pad value: {text_pad_value}"
        )
        logger.debug(f"Original text (start): '{text[:100]}...'")

        # Convert text to bytes and replace special speaker tokens
        byte_text = text.encode("utf-8")
        # Assuming Dia uses byte values 1 and 2 for S1/S2 based on original code context
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)  # List of integer byte values
        logger.debug(
            f"Text tokens after byte conversion (first 10): {text_tokens[:10]}"
        )

        # Pad or truncate sequence
        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            if current_len > max_len:
                logger.warning(
                    f"Input text length ({current_len}) exceeds max length ({max_len}). Truncating."
                )
                text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            logger.debug(f"Padding text input with {padding_needed} pad tokens.")
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        # Convert to tensors and add batch dimension [1, SeqLen]
        src_tokens = (
            torch.from_numpy(padded_text_np)
            .to(torch.long)
            .to(self.target_device)
            .unsqueeze(0)
        )
        src_positions = (
            torch.arange(max_len, device=self.target_device).to(torch.long).unsqueeze(0)
        )

        # Create padding mask (True where token is NOT the pad value)
        src_padding_mask = src_tokens != text_pad_value  # Shape [1, SeqLen]

        # Create attention mask for the encoder (non-causal self-attention)
        # Needs shape [B, 1, Tq, Tk] -> [1, 1, SeqLen, SeqLen]
        enc_self_attn_mask = self._create_attn_mask(
            src_padding_mask, src_padding_mask, is_causal=False
        )

        logger.debug(f"Prepared src_tokens shape: {src_tokens.shape}")
        logger.debug(f"Prepared src_positions shape: {src_positions.shape}")
        logger.debug(
            f"Prepared src_padding_mask shape: {src_padding_mask.shape} (True means non-padding)"
        )
        logger.debug(f"Prepared enc_self_attn_mask shape: {enc_self_attn_mask.shape}")

        return src_tokens, src_positions, src_padding_mask, enc_self_attn_mask

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_cfg_filter: bool = True,
        use_torch_compile: bool = False,  # Default to False for broader compatibility
        cfg_filter_top_k: int = 35,
        audio_prompt_path: str | None = None,
    ) -> np.ndarray:
        """
        Generates audio waveform from a text prompt, optionally conditioned on an audio prompt.

        Args:
            text: The input text string. For dialogue, use [S1]/[S2] markers.
                  For voice cloning, prepend the transcript of the audio prompt.
            max_tokens: Maximum number of audio tokens (frames) to generate. Defaults to config value.
            cfg_scale: Classifier-Free Guidance scale. Higher values increase adherence to text.
            temperature: Sampling temperature. Higher values increase randomness.
            top_p: Nucleus sampling probability. Filters vocabulary during sampling.
            use_cfg_filter: Whether to apply Top-K filtering based on CFG logits.
            use_torch_compile: If True, attempts to compile the decoder step for potential speedup.
            cfg_filter_top_k: The 'K' value for CFG Top-K filtering.
            audio_prompt_path: Path to an audio file (e.g., WAV, MP3) to use as a voice prompt/clone target.

        Returns:
            A 1D NumPy array containing the generated audio waveform (float32).
        """
        start_time_gen = time.time()
        logger.info("Starting audio generation...")
        logger.info(f"  Text (start): '{text[:100]}...'")
        logger.info(
            f"  Max tokens: {max_tokens if max_tokens is not None else 'Model Default'}"
        )
        logger.info(f"  CFG Scale: {cfg_scale}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Top P: {top_p}")
        logger.info(f"  Use CFG Filter: {use_cfg_filter}, Top K: {cfg_filter_top_k}")
        logger.info(
            f"  Audio Prompt: {audio_prompt_path if audio_prompt_path else 'None'}"
        )
        logger.info(f"  Use torch.compile: {use_torch_compile}")
        logger.info(f"  Target Device: {self.target_device}")

        # --- Parameter Setup ---
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        # Use model's default audio length if max_tokens not provided
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.data.audio_length
        )
        logger.info(f"  Effective max_tokens for generation: {effective_max_tokens}")

        # Ensure delay pattern is usable
        if not isinstance(delay_pattern, list) or not delay_pattern:
            logger.warning("Delay pattern is invalid or empty. Using default [0].")
            delay_pattern = [
                0
            ] * num_channels  # Fallback, though config should provide default

        delay_tensor = torch.tensor(
            delay_pattern, dtype=torch.long, device=self.target_device
        )
        max_delay_pattern = max(delay_pattern) if delay_pattern else 0
        self.model.eval()  # Ensure model is in eval mode

        # --- Prepare Conditional and Unconditional Inputs ---
        logger.info(
            "Preparing text inputs for conditional and unconditional generation..."
        )
        (
            cond_src_BxS,
            cond_src_positions_BxS,
            cond_src_padding_mask_BxS,
            cond_enc_self_attn_mask_Bx1xSxS,
        ) = self._prepare_text_input(text)

        # Create unconditional input (batch of zeros representing padding)
        # Assuming pad value 0 for text based on config default
        unc_src_BxS = torch.full_like(
            cond_src_BxS, fill_value=self.config.data.text_pad_value
        )
        # Batch conditional and unconditional inputs together [2, SeqLen]
        src_BxS = torch.cat([unc_src_BxS, cond_src_BxS], dim=0)
        # Expand other inputs to match batch size 2
        src_positions_BxS = cond_src_positions_BxS.expand(2, -1)
        src_padding_mask_BxS = torch.cat(
            [
                torch.zeros_like(cond_src_padding_mask_BxS[0:1]),
                cond_src_padding_mask_BxS,
            ],
            dim=0,
        )  # Uncond mask is all False (padding)
        # Encoder mask needs to handle the batched input correctly
        # For CFG, typically the unconditional branch attends to nothing useful from text,
        # but the structure needs to be maintained. We can reuse the conditional mask structure,
        # but the actual attention scores will be based on the zeroed-out unconditional input.
        # Alternatively, create a specific mask for the unconditional part if needed.
        # Let's expand the conditional mask for simplicity, assuming the model handles zero inputs appropriately.
        enc_self_attn_mask_Bx1xSxS = cond_enc_self_attn_mask_Bx1xSxS.expand(
            2, -1, -1, -1
        )
        logger.info("Text inputs prepared (batch size 2 for CFG).")

        # --- Encoder Pass ---
        logger.info("Running encoder pass...")
        start_time_enc = time.time()
        # Potentially use autocast for mixed precision if supported and beneficial on device
        # Example: with torch.autocast(device_type=self.target_device.type, dtype=torch.bfloat16 if self.target_device.type == 'cuda' else torch.float32):
        encoder_out = self.model.encoder(
            x_ids=src_BxS,  # Shape [2, S]
            src_positions=src_positions_BxS,  # Shape [2, S]
            deterministic=True,  # No dropout during inference
            attn_mask=enc_self_attn_mask_Bx1xSxS,  # Shape [2, 1, S, S]
        )
        logger.info(
            f"Encoder pass completed in {time.time() - start_time_enc:.3f}s. Output shape: {encoder_out.shape}"
        )  # Shape: [2, S, E]

        # --- Prepare Decoder Inputs & KV Cache ---
        logger.info("Preparing decoder inputs and KV cache...")
        start_time_kv = time.time()
        # 3-1. Precompute Cross-Attention KV Cache (Static) from encoder output
        # This cache is computed once and reused for every decoding step.
        decoder_cross_attention_cache: list[KVCache] = (
            self.model.decoder.precompute_cross_attention_kv(
                effective_max_tokens, encoder_out, src_positions_BxS
            )
        )
        logger.debug(
            f"Precomputed cross-attention KV cache for {len(decoder_cross_attention_cache)} layers."
        )

        # 3-2. Initialize Self-Attention KV Cache (Dynamic, grows with each step)
        decoder_self_attention_cache: list[KVCache] = []
        for i in range(self.model.decoder.num_layers):
            decoder_self_attention_cache.append(
                KVCache(
                    self.config.model.decoder.gqa_query_heads,
                    effective_max_tokens,  # Max length the cache can hold
                    self.config.model.decoder.gqa_head_dim,
                    self.target_device,  # Cache tensors should be on the target device
                )
            )
        logger.debug(
            f"Initialized self-attention KV cache for {len(decoder_self_attention_cache)} layers."
        )
        logger.info(
            f"KV cache preparation completed in {time.time() - start_time_kv:.3f}s."
        )

        # 3-3. Initialize Decoder Start Tokens (BOS)
        # Shape [2, 1, C] (Batch=2 for cond/uncond, T=1 for first step, C=channels)
        generated_tokens_history = torch.full(
            (2, 1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.long,
            device=self.target_device,
        )
        logger.debug(f"Initial decoder input (BOS): {generated_tokens_history.shape}")

        current_step_index = (
            0  # Index of the step we are currently generating (starts at 0)
        )
        prompt_len_inc_bos = 1  # Length of the initial prompt (just BOS initially)

        # 3-4. Handle Audio Prompt (Prefill KV Cache)
        if audio_prompt_path is not None:
            logger.info("Processing audio prompt for prefilling...")
            start_time_prompt = time.time()
            try:
                # Load and potentially resample audio
                audio_prompt_waveform, sr = torchaudio.load(audio_prompt_path)
                logger.debug(
                    f"Loaded audio prompt: {audio_prompt_waveform.shape}, Sample Rate: {sr}"
                )
                if sr != 44100:
                    logger.info(f"Resampling audio prompt from {sr}Hz to 44100Hz")
                    audio_prompt_waveform = torchaudio.functional.resample(
                        audio_prompt_waveform, sr, 44100
                    )
                # Ensure correct shape [B, C, T_audio] and device
                # Assuming DAC expects channels first, add batch dim
                if audio_prompt_waveform.ndim == 1:  # Mono
                    audio_prompt_waveform = audio_prompt_waveform.unsqueeze(
                        0
                    )  # Add channel dim
                audio_prompt_waveform = audio_prompt_waveform.unsqueeze(0).to(
                    self.target_device
                )  # Add batch dim

                # Encode audio prompt to codes using DAC
                logger.info("Encoding audio prompt to codes using DAC...")
                if self.dac_model is None:
                    raise RuntimeError(
                        "DAC model not loaded, required for audio prompt."
                    )
                # audio_to_codebook returns [B, T_codes, C]
                audio_prompt_codes = audio_to_codebook(
                    self.dac_model, audio_prompt_waveform, data_config=self.config.data
                )  # Shape [1, T_codes, C]
                logger.info(
                    f"Encoded audio prompt to codes: {audio_prompt_codes.shape}"
                )

                # Concatenate BOS tokens with prompt codes
                # Expand prompt codes to batch size 2 (for cond/uncond)
                generated_tokens_history = torch.cat(
                    [generated_tokens_history, audio_prompt_codes.expand(2, -1, -1)],
                    dim=1,
                )  # Shape [2, 1 + T_codes, C]
                logger.debug(
                    f"Decoder input history after prompt concatenation: {generated_tokens_history.shape}"
                )

                prefill_len = generated_tokens_history.shape[
                    1
                ]  # Length including BOS + prompt
                prompt_len_inc_bos = prefill_len
                logger.info(f"Prefilling KV cache with length {prefill_len}...")

                # Prepare inputs for prefill forward pass
                prefill_tgt_pos = (
                    torch.arange(prefill_len, device=self.target_device)
                    .unsqueeze(0)
                    .expand(2, -1)
                )  # Shape [2, T_prefill]
                # Padding mask based on actual tokens (BOS and prompt codes are not PAD)
                # Shape [2, T_prefill] (True where not PAD)
                prefill_tgt_padding_mask = (
                    generated_tokens_history != audio_pad_value
                ).any(dim=2)

                # Create attention masks for prefill
                # Shape [2, 1, T_prefill, T_prefill]
                prefill_self_attn_mask = self._create_attn_mask(
                    prefill_tgt_padding_mask,
                    prefill_tgt_padding_mask,
                    is_causal=True,
                )
                # Shape [2, 1, T_prefill, S]
                prefill_cross_attn_mask = self._create_attn_mask(
                    prefill_tgt_padding_mask,
                    src_padding_mask_BxS,
                    is_causal=False,
                )

                # Run forward pass through decoder to fill the self-attention KV cache
                # We discard the logits from prefill
                _ = self.model.decoder.forward(
                    tgt_ids_BxTxC=generated_tokens_history,  # Pass the full history [2, T_prefill, C]
                    encoder_out=encoder_out,
                    tgt_positions=prefill_tgt_pos,
                    src_positions=src_positions_BxS,
                    deterministic=True,
                    self_attn_mask=prefill_self_attn_mask,
                    cross_attn_mask=prefill_cross_attn_mask,
                    self_attention_cache=decoder_self_attention_cache,  # Pass cache to be filled
                    cross_attention_cache=decoder_cross_attention_cache,  # Pass precomputed cache
                    # prefill=True # Pass prefill flag if decoder layer uses it
                )

                # Update the current step index. The next token to generate is at index prefill_len.
                current_step_index = prefill_len
                logger.info(
                    f"KV cache prefilled in {time.time() - start_time_prompt:.3f}s. Next step index: {current_step_index}"
                )

            except Exception as e:
                logger.error(f"Error processing audio prompt: {e}", exc_info=True)
                raise RuntimeError("Failed to process audio prompt") from e

        # --- Autoregressive Generation Loop ---
        logger.info("Starting autoregressive generation loop...")
        start_time_loop = time.time()

        eos_detected_channel_0 = False
        eos_countdown = -1  # Countdown after EOS detected in channel 0
        extra_steps_after_eos = (
            30  # Generate a few extra steps for delay pattern completion
        )

        # Pre-allocate tensor for storing *newly* generated tokens for efficiency
        # We already have the prompt in generated_tokens_history
        num_steps_to_generate = effective_max_tokens
        newly_generated_tokens = torch.full(
            (2, num_steps_to_generate, num_channels),
            fill_value=audio_pad_value,  # Fill with pad initially
            dtype=torch.long,
            device=self.target_device,
        )
        logger.debug(
            f"Allocated tensor for newly generated tokens: {newly_generated_tokens.shape}"
        )

        # --- Compile decode_step if requested ---
        decode_step_fn = self.model.decoder.decode_step
        if use_torch_compile:
            logger.info("Compiling decoder step function with torch.compile...")
            try:
                # Experiment with modes: "default", "reduce-overhead", "max-autotune"
                decode_step_fn = torch.compile(decode_step_fn, mode="reduce-overhead")
                logger.info("Decoder step function compiled.")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Using eager mode.", exc_info=True
                )

        # --- Prepare static cross-attention mask for single-step decoding ---
        # Query mask is always [B, 1] (True, as generated tokens are not PAD)
        step_tgt_padding_mask = torch.ones(
            (2, 1), dtype=torch.bool, device=self.target_device
        )
        # Shape [2, 1, 1, S]
        step_decoder_cross_attn_mask = self._create_attn_mask(
            step_tgt_padding_mask,
            src_padding_mask_BxS,
            is_causal=False,
        )

        # --- Generation Loop ---
        steps_taken = 0
        for step_offset in range(num_steps_to_generate):
            # Absolute step index considering prompt length
            current_absolute_step = current_step_index + step_offset

            # Get the token IDs for the *previous* step to predict the current one
            # Shape [2, 1, C]
            # If step_offset is 0, use the last token from the prompt history
            if step_offset == 0:
                input_token_ids = generated_tokens_history[:, -1, :].unsqueeze(1)
            else:
                # Use the token generated in the previous iteration of this loop
                input_token_ids = newly_generated_tokens[
                    :, step_offset - 1, :
                ].unsqueeze(1)

            # Position ID for the current absolute step
            # Shape [2, 1]
            tgt_pos_Bx1 = torch.full(
                (2, 1),
                fill_value=current_absolute_step,
                dtype=torch.long,
                device=self.target_device,
            )

            # --- Call Decoder Step ---
            # self_attn_mask is None because KV cache handles causality implicitly in single-step decoding
            logits_Bx1xCxV, new_self_kv_cache_list = decode_step_fn(
                tgt_ids_Bx1xC=input_token_ids,
                tgt_pos_Bx1=tgt_pos_Bx1,
                encoder_out=encoder_out,
                self_attn_mask=None,
                cross_attn_mask=step_decoder_cross_attn_mask,
                self_attention_cache=decoder_self_attention_cache,
                cross_attention_cache=decoder_cross_attention_cache,
            )  # Logits shape: [2, 1, C, V]

            # --- Update Self-Attention KV Cache ---
            for i, layer_cache in enumerate(decoder_self_attention_cache):
                if (
                    new_self_kv_cache_list
                    and i < len(new_self_kv_cache_list)
                    and new_self_kv_cache_list[i] is not None
                ):
                    # new_self_kv_cache_list[i] is a tuple (k_tensor, v_tensor) for the current step
                    # k_tensor shape: [2, NumHeads, 1, HeadDim]
                    # v_tensor shape: [2, NumHeads, 1, HeadDim]
                    layer_cache.update_cache(
                        new_self_kv_cache_list[i][0], new_self_kv_cache_list[i][1]
                    )
                else:
                    logger.warning(
                        f"Missing KV cache update for layer {i} at step {current_absolute_step}"
                    )

            # --- Sampling ---
            V = self.config.model.tgt_vocab_size
            # Get logits for the generated step [2, C, V]
            logits_last_BxCxV = logits_Bx1xCxV.squeeze(1)

            # Separate conditional and unconditional logits
            uncond_logits_CxV = logits_last_BxCxV[0, :, :]  # Shape [C, V]
            cond_logits_CxV = logits_last_BxCxV[1, :, :]  # Shape [C, V]

            # Apply Classifier-Free Guidance (CFG)
            cfg_logits_CxV = cond_logits_CxV + cfg_scale * (
                cond_logits_CxV - uncond_logits_CxV
            )  # Shape [C, V]

            # --- Prevent sampling PAD/EOS/BOS tokens inappropriately ---
            logits_for_sampling_CxV = (
                cfg_logits_CxV.clone()
            )  # Clone to avoid modifying original logits
            logits_for_sampling_CxV[:, audio_pad_value] = -torch.inf  # Never sample PAD
            logits_for_sampling_CxV[:, audio_bos_value] = (
                -torch.inf
            )  # Never sample BOS after start
            # Allow EOS only if not already detected or in countdown
            if eos_detected_channel_0 and eos_countdown <= 0:
                logits_for_sampling_CxV[:, audio_eos_value] = -torch.inf

            # --- Sample the next token for each channel ---
            pred_C = _sample_next_token(
                logits_for_sampling_CxV.float(),  # Ensure float32 for sampling stability
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=use_cfg_filter,
                cfg_filter_top_k=cfg_filter_top_k,
            )  # Shape [C]

            # --- Handle Delay Pattern (Only if no audio prompt was given) ---
            # If there's no prompt, the first few tokens should be BOS according to delay
            # generation_step_index is how many tokens generated *after* prompt/initial BOS
            generation_step_index = step_offset
            if audio_prompt_path is None:
                is_before_delay = generation_step_index < delay_tensor  # Shape [C]
                pred_C = torch.where(
                    is_before_delay,
                    torch.tensor(
                        audio_bos_value, device=self.target_device, dtype=torch.long
                    ),
                    pred_C,
                )

            # --- Store the predicted token in the newly_generated_tokens tensor ---
            newly_generated_tokens[:, step_offset, :] = pred_C.unsqueeze(0).expand(
                2, -1
            )

            steps_taken += 1  # Increment steps taken in this loop

            # --- EOS Handling ---
            if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
                logger.info(
                    f"EOS token detected in channel 0 at step {current_absolute_step}. Starting countdown."
                )
                eos_detected_channel_0 = True
                eos_countdown = extra_steps_after_eos

            if eos_countdown > 0:
                step_after_eos = extra_steps_after_eos - eos_countdown
                logger.debug(
                    f"EOS countdown: {eos_countdown}, Step after EOS: {step_after_eos}"
                )
                # Modify the token *just generated* if needed for EOS/PAD forcing
                current_new_tokens = newly_generated_tokens[
                    :, step_offset, :
                ]  # Shape [2, C]
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        logger.debug(
                            f"  Forcing EOS in channel {i} at step {current_absolute_step}"
                        )
                        current_new_tokens[:, i] = audio_eos_value
                    elif step_after_eos > d:
                        logger.debug(
                            f"  Forcing PAD in channel {i} at step {current_absolute_step}"
                        )
                        current_new_tokens[:, i] = audio_pad_value
                # Put the potentially modified tokens back
                newly_generated_tokens[:, step_offset, :] = current_new_tokens

                eos_countdown -= 1
                if eos_countdown == 0:
                    logger.info(
                        f"EOS countdown finished at step {current_absolute_step}. Stopping generation."
                    )
                    break  # Stop generation loop

            # Check if we reached the max *new* tokens requested
            if steps_taken >= num_steps_to_generate:
                logger.info(
                    f"Reached max generation steps ({num_steps_to_generate}). Stopping."
                )
                break

        logger.info(
            f"Autoregressive loop finished after {steps_taken} steps in {time.time() - start_time_loop:.3f}s."
        )

        # --- Extract Generated Codes ---
        # Get the conditional generation result (index 1) from the *newly* generated tokens
        # Only take the number of steps actually taken
        final_new_codes = newly_generated_tokens[
            1, :steps_taken, :
        ]  # Shape [T_generated, C]
        logger.info(f"Extracted newly generated codes shape: {final_new_codes.shape}")

        # --- Convert Codes to Audio using DAC ---
        logger.info("Converting generated codes to audio using DAC...")
        start_time_decode = time.time()
        if self.dac_model is None:
            raise RuntimeError("DAC model not loaded, required for audio decoding.")

        # codebook_to_audio expects codes shape [C, T]
        generated_codes_CxT = final_new_codes.transpose(0, 1)  # Shape [C, T_generated]

        if generated_codes_CxT.numel() == 0:
            logger.warning("No new codes were generated. Returning empty audio.")
            return np.array([], dtype=np.float32)

        # Call the decoding function (handles delay reversal and DAC decoding)
        audio_waveform = codebook_to_audio(
            generated_codes_CxT,
            self.dac_model,
            delay_pattern,
            B=1,  # Batch size for decoding is 1
            T=generated_codes_CxT.shape[1],  # Pass the actual length of generated codes
            C=num_channels,
        )  # Returns shape [1, T_audio] or [T_audio]

        # Ensure output is a 1D numpy array on CPU
        final_audio_np = audio_waveform.squeeze().cpu().numpy()
        logger.info(
            f"Audio decoding completed in {time.time() - start_time_decode:.3f}s. Output shape: {final_audio_np.shape}"
        )
        logger.info(f"Total generation time: {time.time() - start_time_gen:.3f}s")

        return final_audio_np
