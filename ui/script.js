// ui/script.js

document.addEventListener('DOMContentLoaded', function () {
    // --- Global Flags ---
    let isGenerating = false;
    let isGenerationCancelled = false;
    let wavesurfer = null; // Global wavesurfer instance

    // --- Element Selectors ---
    const ttsForm = document.getElementById('tts-form');
    const textArea = document.getElementById('text');
    const charCount = document.getElementById('char-count');
    const voiceModeRadios = document.querySelectorAll('input[name="voice_mode"]');
    const cloneOptionsDiv = document.getElementById('clone-options');
    const cloneReferenceSelect = document.getElementById('clone_reference_select');
    const cloneLoadButton = document.getElementById('clone-load-button'); // New ID
    const cloneFileInput = document.getElementById('clone-file-input'); // New ID
    const generateBtn = document.getElementById('generate-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const loadingStatus = document.getElementById('loading-status'); // New element for status
    const loadingCancelBtn = document.getElementById('loading-cancel-btn'); // New ID
    const notificationArea = document.getElementById('notification-area');
    const audioPlayerContainer = document.getElementById('audio-player-container');
    const configSaveBtn = document.getElementById('save-config-btn');
    const configRestartBtn = document.getElementById('restart-server-btn');
    const configStatus = document.getElementById('config-status');
    const genDefaultsSaveBtn = document.getElementById('save-gen-defaults-btn'); // New ID
    const genDefaultsStatus = document.getElementById('gen-defaults-status'); // New ID
    const themeToggleButton = document.getElementById('theme-toggle-btn'); // New ID
    const themeIconLight = document.getElementById('theme-icon-light'); // New ID
    const themeIconDark = document.getElementById('theme-icon-dark'); // New ID
    const presetsContainer = document.getElementById('presets-container'); // New ID

    // --- Initial Setup ---

    // Character counter
    function updateCharCount() {
        if (textArea && charCount) {
            charCount.textContent = textArea.value.length;
        }
    }
    if (textArea) {
        textArea.addEventListener('input', updateCharCount);
        updateCharCount(); // Initial count
    }

    // Toggle Clone Options Visibility & Required Attribute
    function toggleCloneOptions() {
        const selectedMode = document.querySelector('input[name="voice_mode"]:checked')?.value;
        if (cloneOptionsDiv && cloneReferenceSelect && cloneLoadButton) {
            if (selectedMode === 'clone') {
                cloneOptionsDiv.classList.remove('hidden');
                cloneReferenceSelect.required = true;
                cloneLoadButton.classList.remove('hidden');
            } else {
                cloneOptionsDiv.classList.add('hidden');
                cloneReferenceSelect.required = false;
                // cloneReferenceSelect.value = 'none'; // Don't reset if user might switch back
                cloneLoadButton.classList.add('hidden');
            }
        }
    }
    voiceModeRadios.forEach(radio => radio.addEventListener('change', toggleCloneOptions));
    toggleCloneOptions(); // Initial check

    // Update slider value displays dynamically
    const sliders = [
        { id: 'speed_factor', valueId: 'speed_factor_value' },
        { id: 'cfg_scale', valueId: 'cfg_scale_value' },
        { id: 'temperature', valueId: 'temperature_value' },
        { id: 'top_p', valueId: 'top_p_value' },
        { id: 'cfg_filter_top_k', valueId: 'cfg_filter_top_k_value' },
    ];
    sliders.forEach(sliderInfo => {
        const slider = document.getElementById(sliderInfo.id);
        const valueDisplay = document.getElementById(sliderInfo.valueId);
        if (slider && valueDisplay) {
            // Set initial display from slider's current value (set by template)
            valueDisplay.textContent = slider.value;
            // Add event listener to update display on change
            slider.addEventListener('input', () => valueDisplay.textContent = slider.value);
        }
    });

    // --- Notifications ---
    function showNotification(message, type = 'success', duration = 5000) {
        if (!notificationArea) return;
        // notificationArea.innerHTML = ''; // Clear previous? Or allow multiple? Let's allow multiple for now.
        const colors = {
            success: 'notification-success',
            error: 'notification-error',
            warning: 'notification-warning',
            info: 'notification-info' // Add info style if needed
        };
        const icons = { // SVG icons or classes
            success: '<svg class="h-5 w-5 text-green-500 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" /></svg>',
            error: '<svg class="h-5 w-5 text-red-500 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>',
            warning: '<svg class="h-5 w-5 text-yellow-500 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" /></svg>',
            info: '<svg class="h-5 w-5 text-sky-500 mr-2 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd" /></svg>'
        };

        const notificationDiv = document.createElement('div');
        notificationDiv.className = colors[type] || colors['info']; // Default to info style
        notificationDiv.innerHTML = `${icons[type] || icons['info']} <span class="block sm:inline">${message}</span>`;
        notificationArea.appendChild(notificationDiv);

        // Auto-hide after specified duration
        if (duration > 0) {
            setTimeout(() => {
                notificationDiv.style.transition = 'opacity 0.5s ease-out';
                notificationDiv.style.opacity = '0';
                setTimeout(() => notificationDiv.remove(), 500);
            }, duration);
        }
        return notificationDiv; // Return the element if manual removal is needed
    }

    // --- Presets ---
    function applyPreset(presetData) {
        console.log("Applying preset:", presetData);
        if (!presetData) return;

        // Update text area
        if (textArea && presetData.text !== undefined) {
            textArea.value = presetData.text;
            updateCharCount(); // Update counter
        }

        // Update voice mode
        if (presetData.voice_mode) {
            const radio = document.querySelector(`input[name="voice_mode"][value="${presetData.voice_mode}"]`);
            if (radio) {
                radio.checked = true;
                toggleCloneOptions(); // Update UI based on new mode
            }
        }

        // Update generation parameters
        if (presetData.params) {
            for (const [key, value] of Object.entries(presetData.params)) {
                const slider = document.getElementById(key); // Assumes slider ID matches param key
                const valueDisplay = document.getElementById(`${key}_value`);
                if (slider) {
                    slider.value = value;
                    if (valueDisplay) {
                        valueDisplay.textContent = value; // Update display
                    }
                } else {
                    console.warn(`Slider element not found for preset parameter: ${key}`);
                }
            }
        }
        showNotification(`Preset "${presetData.name}" loaded.`, 'info', 3000);
    }

    // Add event listeners to preset buttons (assuming they exist)
    // Presets data should be available globally, e.g., from template `window.appPresets = {{ presets | tojson }};`
    if (window.appPresets && presetsContainer) {
        window.appPresets.forEach((preset, index) => {
            const button = document.getElementById(`preset-btn-${index}`);
            if (button) {
                button.addEventListener('click', () => applyPreset(preset));
            }
        });
    } else if (presetsContainer) {
        console.warn("Presets data (window.appPresets) not found, preset buttons will not work.");
    }


    // --- Audio Player ---
    function initializeWaveSurfer(audioUrl) {
        if (wavesurfer) {
            wavesurfer.destroy();
        }
        const waveformDiv = document.getElementById('waveform');
        const playBtn = document.getElementById('play-btn');
        const durationSpan = document.getElementById('audio-duration');

        if (!waveformDiv || !playBtn || !durationSpan) {
            console.error("Audio player elements not found in the container.");
            // Clear the container if elements are missing after generation
            if (audioPlayerContainer) audioPlayerContainer.innerHTML = '<p class="text-red-500 dark:text-red-400">Error displaying audio player.</p>';
            return;
        }

        // Ensure button text doesn't wrap
        playBtn.classList.add('whitespace-nowrap', 'flex-shrink-0');
        const downloadLink = document.getElementById('download-link');
        if (downloadLink) downloadLink.classList.add('whitespace-nowrap', 'flex-shrink-0');


        wavesurfer = WaveSurfer.create({
            container: waveformDiv,
            waveColor: document.documentElement.classList.contains('dark') ? '#38bdf8' : '#0ea5e9', // primary-400(dark) / primary-500(light)
            progressColor: document.documentElement.classList.contains('dark') ? '#0284c7' : '#0369a1', // primary-600(dark) / primary-700(light)
            cursorColor: document.documentElement.classList.contains('dark') ? '#a855f7' : '#9333ea', // purple-500(dark) / purple-600(light)
            barWidth: 3,
            barRadius: 3,
            cursorWidth: 1,
            height: 80,
            barGap: 2,
            responsive: true,
            url: audioUrl,
            mediaControls: false, // Use custom controls
            normalize: true,
        });

        wavesurfer.on('ready', () => {
            const duration = wavesurfer.getDuration();
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60);
            durationSpan.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
            playBtn.disabled = false;
            playBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg> Play`;
        });

        wavesurfer.on('play', () => {
            playBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm5-2.25A.75.75 0 0 1 7.75 7h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-.75.75h-4.5a.75.75 0 0 1-.75-.75v-4.5Z" clip-rule="evenodd" /></svg> Pause`;
        });

        wavesurfer.on('pause', () => {
            playBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg> Play`;
        });
        wavesurfer.on('finish', () => {
            playBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-5 h-5 mr-1"><path fill-rule="evenodd" d="M2 10a8 8 0 1 1 16 0 8 8 0 0 1-16 0Zm6.39-2.908a.75.75 0 0 1 .766.027l3.5 2.25a.75.75 0 0 1 0 1.262l-3.5 2.25A.75.75 0 0 1 8 12.25v-4.5a.75.75 0 0 1 .39-.658Z" clip-rule="evenodd" /></svg> Play`;
        });

        playBtn.onclick = () => {
            wavesurfer.playPause();
        };

        // Scroll to the player after initialization
        setTimeout(() => {
            audioPlayerContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100); // Short delay to ensure rendering
    }

    // Initialize player if audio URL is present on initial page load
    // This logic needs to be adapted as the player is now dynamically added
    // We'll call initializeWaveSurfer if the template renders the player container
    const initialAudioUrlElement = document.querySelector('[data-initial-audio-url]');
    if (initialAudioUrlElement && initialAudioUrlElement.dataset.initialAudioUrl) {
        console.log("Initializing WaveSurfer for initially loaded audio.");
        initializeWaveSurfer(initialAudioUrlElement.dataset.initialAudioUrl);
    }


    // --- Form Submission & Cancellation ---
    if (ttsForm) {
        ttsForm.addEventListener('submit', function (event) {
            // Client-side validation
            const text = textArea.value.trim();
            const mode = document.querySelector('input[name="voice_mode"]:checked')?.value;
            const cloneFile = cloneReferenceSelect?.value;

            if (!text) {
                showNotification("Please enter some text.", 'error');
                event.preventDefault(); return;
            }
            if (mode === 'clone' && (!cloneFile || cloneFile === 'none')) {
                showNotification("Please select a reference file for clone mode.", 'error');
                event.preventDefault(); return;
            }

            // Handle cancellation of previous request if Generate is clicked again
            if (isGenerating) {
                console.log("Generate clicked while previous generation in progress. Setting cancel flag.");
                showNotification("Cancelling previous request...", 'warning', 2000);
                isGenerationCancelled = true;
                // We don't actually stop the backend here (Fake Cancel)
                // but the result processing will ignore the previous result.
            }

            // Reset flags and show loading overlay for the new request
            isGenerating = true;
            isGenerationCancelled = false; // Reset cancel flag for the new request
            if (loadingOverlay && generateBtn && loadingCancelBtn) {
                loadingMessage.textContent = 'Generating audio...'; // Initial status
                loadingStatus.textContent = 'Please wait.';
                loadingOverlay.classList.remove('hidden');
                generateBtn.disabled = true;
                generateBtn.classList.add('opacity-50', 'cursor-not-allowed');
                loadingCancelBtn.disabled = false; // Enable cancel button
            }
            // Allow default form submission to proceed
            // The page will reload with results rendered by the template
        });
    }

    // Handle Cancel button click
    if (loadingCancelBtn) {
        loadingCancelBtn.addEventListener('click', () => {
            if (isGenerating) {
                console.log("Cancel button clicked.");
                isGenerationCancelled = true;
                isGenerating = false; // Stop considering it "generating" from UI perspective
                if (loadingOverlay && generateBtn) {
                    loadingOverlay.classList.add('hidden'); // Hide overlay
                    generateBtn.disabled = false; // Re-enable generate button
                    generateBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                }
                showNotification("Generation cancelled by user.", 'info');
                // Note: Backend request continues, but result will be ignored on page reload/update
            }
        });
    }

    // --- Result Handling (on page load after form submission) ---
    // This logic runs every time the page loads. We check if specific elements
    // indicating a successful generation are present.
    const outputUrlElement = document.getElementById('output-file-url-data'); // Need to add this element in HTML
    if (outputUrlElement && outputUrlElement.dataset.url) {
        const outputUrl = outputUrlElement.dataset.url;
        console.log("Page loaded with generation result:", outputUrl);

        if (isGenerationCancelled) {
            console.log("Generation was cancelled, ignoring result.");
            showNotification("Previous generation was cancelled.", "warning");
            // Reset flag after checking
            isGenerationCancelled = false;
        } else {
            console.log("Processing successful generation result.");
            // The audio player structure should be rendered by the template.
            // We just need to initialize wavesurfer for it.
            initializeWaveSurfer(outputUrl);
        }
    }
    // Always reset generating flag on page load, as any active generation is now finished or irrelevant
    isGenerating = false;
    if (generateBtn) { // Re-enable button if page reloads for any reason
        generateBtn.disabled = false;
        generateBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }


    // --- Configuration Management ---
    async function updateConfigStatus(button, statusElement, message, success = true, duration = 5000) {
        const successClass = 'text-green-500 dark:text-green-400';
        const errorClass = 'text-red-500 dark:text-red-400';
        const savingClass = 'text-yellow-500 dark:text-yellow-400';

        statusElement.textContent = message;
        statusElement.className = `text-xs ml-2 ${success ? successClass : (message.startsWith('Saving') || message.startsWith('Restarting') ? savingClass : errorClass)}`;
        statusElement.classList.remove('hidden');
        if (button) button.disabled = true; // Disable button while processing

        // Clear status after duration, re-enable button
        if (duration > 0) {
            setTimeout(() => {
                statusElement.classList.add('hidden');
                if (button) button.disabled = false;
            }, duration);
        }
    }

    // Save Server Configuration
    if (configSaveBtn) {
        configSaveBtn.addEventListener('click', async () => {
            const configData = {};
            document.querySelectorAll('#server-config-form input[name]').forEach(input => { // Assume inputs are within a form/div
                configData[input.name] = input.value;
            });

            updateConfigStatus(configSaveBtn, configStatus, 'Saving...', true, 0); // Indefinite until success/error

            try {
                const response = await fetch('/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(configData)
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save');

                updateConfigStatus(configSaveBtn, configStatus, result.message, true);
                if (configRestartBtn) configRestartBtn.classList.remove('hidden'); // Show restart button

            } catch (error) {
                console.error('Error saving server config:', error);
                updateConfigStatus(configSaveBtn, configStatus, `Error: ${error.message}`, false);
            }
        });
    }

    // Restart Server
    if (configRestartBtn) {
        configRestartBtn.addEventListener('click', async () => {
            configRestartBtn.disabled = true;
            configRestartBtn.innerHTML = `
               <svg class="animate-spin h-5 w-5 mr-1 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                 <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
               </svg>
               Restarting...`;
            updateConfigStatus(configRestartBtn, configStatus, 'Restarting...', true, 0); // Indefinite

            try {
                const response = await fetch('/restart_server', { method: 'POST' });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to trigger restart');

                updateConfigStatus(configRestartBtn, configStatus, result.message + " Page will attempt reload.", true, 15000); // Show longer
                // Show main loading overlay during restart check
                if (loadingOverlay) {
                    loadingMessage.textContent = 'Server restarting...';
                    loadingStatus.textContent = 'Waiting for server to respond...';
                    loadingCancelBtn.disabled = true; // Disable cancel during restart
                    loadingOverlay.classList.remove('hidden');
                }

                // Poll for server readiness
                let attempts = 0;
                const maxAttempts = 45; // Wait up to 45 seconds
                function checkServerReady() {
                    attempts++;
                    console.log(`Checking server readiness (Attempt ${attempts}/${maxAttempts})...`);
                    loadingStatus.textContent = `Waiting for server... (${attempts}/${maxAttempts})`;
                    fetch('/health?cache=' + Date.now(), { cache: 'no-store', headers: { 'pragma': 'no-cache' } })
                        .then(res => {
                            if (res.ok) {
                                console.log("Server is ready. Reloading page.");
                                window.location.reload(true); // Force reload from server
                            } else if (attempts < maxAttempts) {
                                setTimeout(checkServerReady, 1000); // Check again in 1 second
                            } else {
                                throw new Error('Server did not become ready after restart.');
                            }
                        })
                        .catch(() => {
                            if (attempts < maxAttempts) {
                                setTimeout(checkServerReady, 1000); // Check again on connection error
                            } else {
                                throw new Error('Server did not respond after restart.');
                            }
                        });
                }
                setTimeout(checkServerReady, 3000); // Start checking after 3 seconds

            } catch (error) {
                console.error('Error restarting server:', error);
                updateConfigStatus(configRestartBtn, configStatus, `Restart Error: ${error.message}`, false);
                configRestartBtn.disabled = false; // Re-enable button on error
                configRestartBtn.innerHTML = `
                 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 mr-1 inline-block"><path stroke-linecap="round" stroke-linejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" /></svg>
                 Restart Server`;
                if (loadingOverlay) loadingOverlay.classList.add('hidden');
            }
        });
    }

    // Save Generation Defaults
    if (genDefaultsSaveBtn) {
        genDefaultsSaveBtn.addEventListener('click', async () => {
            const genParams = {};
            sliders.forEach(s => {
                const slider = document.getElementById(s.id);
                if (slider) genParams[s.id] = slider.value;
            });

            updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, 'Saving...', true, 0);

            try {
                const response = await fetch('/save_generation_defaults', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(genParams)
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.detail || 'Failed to save');
                updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, result.message, true);

            } catch (error) {
                console.error('Error saving generation defaults:', error);
                updateConfigStatus(genDefaultsSaveBtn, genDefaultsStatus, `Error: ${error.message}`, false);
            }
        });
    }

    // --- Reference Audio Upload ---
    if (cloneLoadButton && cloneFileInput && cloneReferenceSelect) {
        cloneLoadButton.addEventListener('click', () => {
            cloneFileInput.click(); // Trigger hidden file input
        });

        cloneFileInput.addEventListener('change', async (event) => {
            const files = event.target.files;
            if (!files || files.length === 0) {
                return; // No files selected
            }

            cloneLoadButton.disabled = true;
            cloneLoadButton.textContent = 'Uploading...';
            showNotification(`Uploading ${files.length} file(s)...`, 'info', 0); // Indefinite

            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }

            try {
                const response = await fetch('/upload_reference', {
                    method: 'POST',
                    body: formData
                    // Content-Type is set automatically for FormData
                });

                const result = await response.json();

                // Clear existing notifications before showing results
                notificationArea.innerHTML = '';

                if (!response.ok) {
                    throw new Error(result.message || `Upload failed with status ${response.status}`);
                }

                // Process results
                if (result.errors && result.errors.length > 0) {
                    result.errors.forEach(err => showNotification(err, 'error'));
                }
                if (result.uploaded_files && result.uploaded_files.length > 0) {
                    showNotification(`Successfully uploaded: ${result.uploaded_files.join(', ')}`, 'success');
                } else if (!result.errors || result.errors.length === 0) {
                    showNotification("Files processed, but no new files were added (might already exist).", 'info');
                }


                // Update dropdown
                const currentSelection = cloneReferenceSelect.value;
                cloneReferenceSelect.innerHTML = '<option value="none">-- Select Reference File --</option>'; // Clear existing options
                result.all_reference_files.forEach(filename => {
                    const option = document.createElement('option');
                    option.value = filename;
                    option.textContent = filename;
                    cloneReferenceSelect.appendChild(option);
                });

                // Select the first newly uploaded file, or keep current selection if still valid
                const firstUploaded = result.uploaded_files ? result.uploaded_files[0] : null;
                if (firstUploaded) {
                    cloneReferenceSelect.value = firstUploaded;
                } else if (result.all_reference_files.includes(currentSelection)) {
                    cloneReferenceSelect.value = currentSelection; // Restore previous valid selection
                } else {
                    cloneReferenceSelect.value = 'none'; // Default if nothing else matches
                }

            } catch (error) {
                console.error('Error uploading reference files:', error);
                showNotification(`Upload Error: ${error.message}`, 'error');
            } finally {
                cloneLoadButton.disabled = false;
                cloneLoadButton.textContent = 'Load';
                cloneFileInput.value = ''; // Reset file input
            }
        });
    }

    // --- Theme Toggle ---
    function applyTheme(theme) {
        if (theme === 'light') {
            document.documentElement.classList.remove('dark');
            if (themeIconLight) themeIconLight.classList.remove('hidden');
            if (themeIconDark) themeIconDark.classList.add('hidden');
        } else {
            document.documentElement.classList.add('dark');
            if (themeIconLight) themeIconLight.classList.add('hidden');
            if (themeIconDark) themeIconDark.classList.remove('hidden');
        }
        // Update wavesurfer colors if it exists
        if (wavesurfer) {
            wavesurfer.setOptions({
                waveColor: theme === 'light' ? '#0ea5e9' : '#38bdf8',
                progressColor: theme === 'light' ? '#0369a1' : '#0284c7',
                cursorColor: theme === 'light' ? '#9333ea' : '#a855f7',
            });
        }
    }

    if (themeToggleButton) {
        // Check localStorage on load
        const savedTheme = localStorage.getItem('theme') || 'dark'; // Default to dark
        applyTheme(savedTheme);

        themeToggleButton.addEventListener('click', () => {
            const isDark = document.documentElement.classList.contains('dark');
            const newTheme = isDark ? 'light' : 'dark';
            applyTheme(newTheme);
            localStorage.setItem('theme', newTheme); // Save preference
        });
    }

}); // End DOMContentLoaded