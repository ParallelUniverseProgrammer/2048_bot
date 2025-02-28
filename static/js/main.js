// Initialize Socket.IO
const socket = io();

// Store the default hyperparameters
const defaultHyperparams = {
    // Learning Parameters
    learning_rate: 0.0003,
    early_lr_multiplier: 1.8,
    warmup_episodes: 25,
    grad_clip: 0.85,
    lr_scheduler_patience: 80,
    lr_scheduler_factor: 0.75,
    
    // Architecture Parameters
    base_dmodel: 192,
    base_nhead: 8,
    base_transformer_layers: 6,
    base_high_level_layers: 1,
    base_dropout: 0.15,
    
    // Reward Function Parameters
    high_tile_bonus: 5.5,
    ineffective_penalty: 0.15,
    reward_scaling: 0.12,
    time_factor_constant: 50.0,
    novelty_bonus: 4.0,
    high_tile_threshold: 512,
    pattern_diversity_bonus: 2.0,
    strategy_shift_bonus: 1.0,
    
    // Exploration Parameters
    use_temperature_annealing: true,
    initial_temperature: 1.4,
    final_temperature: 0.8,
    temperature_decay: 0.99995,
    
    // Training Parameters
    base_batch_size: 20,
    model_save_interval: 200
};

// Current hyperparameters (copy of default initially)
const hyperparams = {...defaultHyperparams};

// DOM Elements - Main controls
const trainingButton = document.getElementById('start-training');
const stopTrainingButton = document.getElementById('stop-training');
const watchButton = document.getElementById('start-watch');
const trainingPanel = document.getElementById('training-panel');
const watchPanel = document.getElementById('watch-panel');
const gameBoard = document.getElementById('game-board');
const gameScore = document.getElementById('game-score');
const watchBestTile = document.getElementById('watch-best-tile');
const moveCount = document.getElementById('move-count');
const themeSelector = document.getElementById('theme-selector');
const toggleHardware = document.getElementById('toggle-hardware');
const hardwarePanel = document.getElementById('hardware-panel');
const toggleHyperparams = document.getElementById('toggle-hyperparams');
const hyperparamsContent = document.getElementById('hyperparams-content');
const applyHyperparamsButton = document.getElementById('apply-hyperparams');
const resetHyperparamsButton = document.getElementById('reset-hyperparams');
const tabButtons = document.querySelectorAll('.tab-button');
const tabContents = document.querySelectorAll('.tab-content');
const toastContainer = document.getElementById('toast-container');

// Checkpoint panel elements
const toggleCheckpoint = document.getElementById('toggle-checkpoint');
const checkpointPanel = document.getElementById('checkpoint-panel');
const checkpointStatusLoading = document.getElementById('checkpoint-status-loading');
const checkpointStatusEmpty = document.getElementById('checkpoint-status-empty');
const checkpointStatusError = document.getElementById('checkpoint-status-error');
const checkpointStatusLoaded = document.getElementById('checkpoint-status-loaded');
const checkpointErrorMessage = document.getElementById('checkpoint-error-message');
const checkpointCreated = document.getElementById('checkpoint-created');
const checkpointAge = document.getElementById('checkpoint-age');
const checkpointTrainingTime = document.getElementById('checkpoint-training-time');
const checkpointEpisodes = document.getElementById('checkpoint-episodes');
const checkpointReward = document.getElementById('checkpoint-reward');
const checkpointBestTile = document.getElementById('checkpoint-best-tile');
const checkpointSize = document.getElementById('checkpoint-size');
const downloadCheckpoint = document.getElementById('download-checkpoint');
const deleteCheckpoint = document.getElementById('delete-checkpoint');

// Confirmation dialog elements
const confirmationDialog = document.getElementById('confirmation-dialog');
const dialogTitle = document.getElementById('dialog-title');
const dialogMessage = document.getElementById('dialog-message');
const dialogConfirm = document.getElementById('dialog-confirm');
const dialogCancel = document.getElementById('dialog-cancel');
const dialogClose = document.getElementById('dialog-close');

// Chart containers
const rewardChart = document.getElementById('reward-chart').getContext('2d');
const maxTileChart = document.getElementById('max-tile-chart').getContext('2d');
const lossChart = document.getElementById('loss-chart').getContext('2d');
const movesChart = document.getElementById('moves-chart').getContext('2d');

// Hardware monitoring elements
const cpuUsage = document.getElementById('cpu-usage');
const ramUsage = document.getElementById('ram-usage');
const gpuInfo = document.getElementById('gpu-info');

// Training stats elements
const totalEpisodesElem = document.getElementById('total-episodes');
const avgBatchRewardElem = document.getElementById('avg-batch-reward');
const recentAvgRewardElem = document.getElementById('recent-avg-reward');
const bestAvgRewardElem = document.getElementById('best-avg-reward');
const avgEpisodeLengthElem = document.getElementById('avg-episode-length');
const batchMaxTileElem = document.getElementById('batch-max-tile');
const bestTileElem = document.getElementById('best-tile');
const currentLrElem = document.getElementById('current-lr');

// State variables
let currentMode = null;
let charts = {};
let lastCheckpointTime = 0;
let chartUpdateTimeout = null;
let chartData = {
    rewards: [],
    maxTiles: [],
    losses: [],
    moves: []
};

// Link all hyperparameter sliders to their number inputs and gather references
const hyperparamControls = {};

function setupHyperparamControls() {
    // Learning parameters
    setupRangeInput('learning-rate', 'learning_rate');
    setupRangeInput('early-lr-mult', 'early_lr_multiplier');
    setupRangeInput('warmup-episodes', 'warmup_episodes');
    setupRangeInput('grad-clip', 'grad_clip');
    setupRangeInput('lr-scheduler-patience', 'lr_scheduler_patience');
    setupRangeInput('lr-scheduler-factor', 'lr_scheduler_factor');
    
    // Architecture parameters
    setupRangeInput('base-dmodel', 'base_dmodel');
    setupRangeInput('base-nhead', 'base_nhead');
    setupRangeInput('base-transformer-layers', 'base_transformer_layers');
    setupRangeInput('base-high-level-layers', 'base_high_level_layers');
    setupRangeInput('base-dropout', 'base_dropout');
    
    // Reward parameters
    setupRangeInput('high-tile-bonus', 'high_tile_bonus');
    setupRangeInput('ineffective-penalty', 'ineffective_penalty');
    setupRangeInput('reward-scaling', 'reward_scaling');
    setupRangeInput('time-factor-constant', 'time_factor_constant');
    setupRangeInput('novelty-bonus', 'novelty_bonus');
    setupSelectInput('high-tile-threshold', 'high_tile_threshold');
    setupRangeInput('pattern-diversity-bonus', 'pattern_diversity_bonus');
    setupRangeInput('strategy-shift-bonus', 'strategy_shift_bonus');
    
    // Exploration parameters
    setupToggleInput('temperature-annealing', 'use_temperature_annealing');
    setupRangeInput('initial-temperature', 'initial_temperature');
    setupRangeInput('final-temperature', 'final_temperature');
    setupRangeInput('temperature-decay', 'temperature_decay');
    
    // Training parameters
    setupRangeInput('base-batch-size', 'base_batch_size');
    setupRangeInput('model-save-interval', 'model_save_interval');
}

// Setup range input (slider) and link with number input
function setupRangeInput(inputId, paramName) {
    const slider = document.getElementById(inputId);
    const valueInput = document.getElementById(`${inputId}-value`);
    
    if (!slider || !valueInput) return;
    
    hyperparamControls[paramName] = { slider, valueInput };
    
    // Update number input when slider changes
    slider.addEventListener('input', function() {
        valueInput.value = this.value;
        hyperparams[paramName] = parseFloat(this.value);
    });
    
    // Update slider when number input changes
    valueInput.addEventListener('input', function() {
        slider.value = this.value;
        hyperparams[paramName] = parseFloat(this.value);
    });
    
    // Initialize with default value
    slider.value = defaultHyperparams[paramName];
    valueInput.value = defaultHyperparams[paramName];
}

// Setup select dropdown
function setupSelectInput(inputId, paramName) {
    const select = document.getElementById(inputId);
    
    if (!select) return;
    
    hyperparamControls[paramName] = { select };
    
    select.addEventListener('change', function() {
        hyperparams[paramName] = parseInt(this.value);
    });
    
    // Initialize with default value
    select.value = defaultHyperparams[paramName];
}

// Setup toggle switch
function setupToggleInput(inputId, paramName) {
    const toggle = document.getElementById(inputId);
    
    if (!toggle) return;
    
    hyperparamControls[paramName] = { toggle };
    
    toggle.addEventListener('change', function() {
        hyperparams[paramName] = this.checked;
    });
    
    // Initialize with default value
    toggle.checked = defaultHyperparams[paramName];
}

// Reset hyperparameters to defaults
function resetHyperparameters() {
    // Reset the hyperparams object
    Object.assign(hyperparams, defaultHyperparams);
    
    // Update all UI controls
    for (const [paramName, controls] of Object.entries(hyperparamControls)) {
        const defaultValue = defaultHyperparams[paramName];
        
        if (controls.slider && controls.valueInput) {
            // Range input with linked number input
            controls.slider.value = defaultValue;
            controls.valueInput.value = defaultValue;
        } else if (controls.select) {
            // Select dropdown
            controls.select.value = defaultValue;
        } else if (controls.toggle) {
            // Toggle switch
            controls.toggle.checked = defaultValue;
        }
    }
    
    showToast('Hyperparameters reset to defaults', 'info');
}

// Apply hyperparameters to the model
function applyHyperparameters() {
    // Send hyperparameters to the server
    socket.emit('set_hyperparams', hyperparams);
    
    // Show loading indicator
    showLoadingIndicator(true, 'Applying hyperparameters...');
    
    // Wait for confirmation from server
    setTimeout(() => {
        showLoadingIndicator(false);
        showToast('Hyperparameters applied successfully', 'success');
    }, 1000);
}

// Theme management
function setTheme(theme) {
    const body = document.body;
    if (theme === 'light') {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
    } else {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
    }
    
    // Update chart colors
    updateChartTheme(theme);
    
    // Save theme preference to localStorage
    localStorage.setItem('theme', theme);
}

// Update chart colors based on theme
function updateChartTheme(theme) {
    if (!charts.reward) return;
    
    const isDark = theme === 'dark';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
    
    const updateChartColors = (chart) => {
        // Update grid colors
        chart.options.scales.x.grid.color = gridColor;
        chart.options.scales.y.grid.color = gridColor;
        
        // Update text colors
        chart.options.scales.x.ticks.color = textColor;
        chart.options.scales.y.ticks.color = textColor;
        chart.options.scales.x.title.color = textColor;
        chart.options.plugins.legend.labels.color = textColor;
        chart.options.plugins.title.color = textColor;
        
        chart.update();
    };
    
    // Update all charts
    updateChartColors(charts.reward);
    updateChartColors(charts.maxTile);
    updateChartColors(charts.loss);
    updateChartColors(charts.moves);
}

// Show a toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'info';
    if (type === 'success') icon = 'check_circle';
    if (type === 'error') icon = 'error';
    if (type === 'warning') icon = 'warning';
    
    toast.innerHTML = `
        <i class="material-icons">${icon}</i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 300);
    }, 3000);
}

// Initialize Charts
function initCharts() {
    // Get theme-appropriate colors
    const isDark = document.body.classList.contains('dark-theme');
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDark ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.7)';
    
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Disable animations for faster rendering
        },
        scales: {
            x: {
                grid: {
                    color: gridColor
                },
                ticks: {
                    color: textColor,
                    maxTicksLimit: 5 // Limit ticks for better performance
                },
                title: {
                    display: true,
                    text: 'Episode',
                    color: textColor
                }
            },
            y: {
                grid: {
                    color: gridColor
                },
                ticks: {
                    color: textColor
                }
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: textColor
                }
            },
            tooltip: {
                enabled: false // Disable tooltips for better performance
            }
        }
    };

    // Create reward chart
    charts.reward = new Chart(rewardChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Reward',
                data: [],
                borderColor: '#6200ee',
                backgroundColor: 'rgba(98, 0, 238, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Rewards',
                    color: textColor
                }
            }
        }
    });

    // Create max tile chart
    charts.maxTile = new Chart(maxTileChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Max Tile',
                data: [],
                borderColor: '#f65e3b',
                backgroundColor: 'rgba(246, 94, 59, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Max Tile',
                    color: textColor
                }
            }
        }
    });

    // Create loss chart
    charts.loss = new Chart(lossChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#cf6679',
                backgroundColor: 'rgba(207, 102, 121, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Loss',
                    color: textColor
                }
            }
        }
    });
    
    // Create moves chart
    charts.moves = new Chart(movesChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Moves',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Moves',
                    color: textColor
                }
            }
        }
    });
}

// Initialize Game Board
function initGameBoard() {
    gameBoard.innerHTML = '';
    for (let i = 0; i < 16; i++) {
        const tile = document.createElement('div');
        tile.className = 'tile tile-0';
        tile.innerText = '';
        gameBoard.appendChild(tile);
    }
}

// Update Game Board
function updateGameBoard(board) {
    const tiles = gameBoard.querySelectorAll('.tile');
    let index = 0;
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const value = board[i][j];
            tiles[index].className = `tile tile-${value}`;
            tiles[index].innerText = value ? value : '';
            index++;
        }
    }
}

// Update all charts with new data - optimized version
function updateCharts(data) {
    // Handle potentially missing baseEpisode
    const baseEpisode = data.episode_base || 0;
    
    // Clear existing data if needed
    if (charts.reward.data.labels.length > 0 && 
        (charts.reward.data.labels[0] !== baseEpisode || 
         charts.reward.data.labels.length !== data.rewards_chart.length)) {
        // Different base or length - reset all charts
        charts.reward.data.labels = [];
        charts.reward.data.datasets[0].data = [];
        charts.maxTile.data.labels = [];
        charts.maxTile.data.datasets[0].data = [];
        charts.loss.data.labels = [];
        charts.loss.data.datasets[0].data = [];
        charts.moves.data.labels = [];
        charts.moves.data.datasets[0].data = [];
    }
    
    // Generate labels based on episode numbers
    const labels = Array.from({ length: data.rewards_chart.length }, 
                             (_, i) => baseEpisode + i);
    
    // Update chart data
    charts.reward.data.labels = labels;
    charts.reward.data.datasets[0].data = data.rewards_chart;
    
    charts.maxTile.data.labels = labels;
    charts.maxTile.data.datasets[0].data = data.max_tile_chart;
    
    if (data.loss_chart && data.loss_chart.length) {
        charts.loss.data.labels = labels;
        charts.loss.data.datasets[0].data = data.loss_chart;
    }
    
    if (data.moves_chart && data.moves_chart.length) {
        charts.moves.data.labels = labels;
        charts.moves.data.datasets[0].data = data.moves_chart;
    }
    
    // Update all charts in one batch
    charts.reward.update();
    charts.maxTile.update();
    charts.loss.update();
    charts.moves.update();
}

// Update hardware info
function updateHardwareInfo(data) {
    cpuUsage.innerHTML = `
        <div class="hardware-value">${data.cpu.usage.toFixed(1)}%</div>
        <div class="hardware-details">${data.cpu.cores} cores</div>
    `;
    
    ramUsage.innerHTML = `
        <div class="hardware-value">${data.ram.used_gb.toFixed(1)} GB</div>
        <div class="hardware-details">${data.ram.percent.toFixed(1)}% of ${data.ram.total_gb.toFixed(1)} GB</div>
    `;
    
    let gpuHtml = '';
    if (data.gpus.length > 0) {
        data.gpus.forEach((gpu, index) => {
            gpuHtml += `
                <div class="hardware-card">
                    <div class="hardware-title"><i class="material-icons">memory</i>GPU ${index}: ${gpu.name}</div>
                    <div class="hardware-value">${gpu.memory_used.toFixed(1)} GB</div>
                    <div class="hardware-details">
                        ${gpu.memory_percent.toFixed(1)}% of ${gpu.memory_total.toFixed(1)} GB
                        <br>${gpu.usage.toFixed(1)}% Utilization
                        <br>${gpu.temperature}°C
                    </div>
                </div>
            `;
        });
    } else {
        gpuHtml = '<div class="hardware-card"><div class="hardware-value">No GPUs detected</div></div>';
    }
    
    gpuInfo.innerHTML = gpuHtml;
}

// Track previous values for highlighting changes
const prevValues = {
    total_episodes: 0,
    avg_batch_reward: 0,
    recent_avg_reward: 0,
    best_avg_reward: 0,
    batch_max_tile: 0,
    best_max_tile: 0,
    avg_batch_moves: 0,
    current_lr: 0
};

// Direct update of all stat fields with highlighting for changes
function updateTrainingStats(data) {
    // Helper to update a stat with highlight effect
    function updateStatWithHighlight(elem, newValue, prevKey, format = null) {
        const formattedValue = format ? format(newValue) : newValue;
        elem.textContent = formattedValue;
        
        // Apply highlight effect if value changed
        if (newValue !== prevValues[prevKey]) {
            // Add highlight class
            elem.classList.remove('highlight');
            void elem.offsetWidth; // Force reflow for animation restart
            elem.classList.add('highlight');
            
            // Update stored value
            prevValues[prevKey] = newValue;
        }
    }
    
    // Update episode count
    if (data.total_episodes !== undefined) {
        updateStatWithHighlight(totalEpisodesElem, data.total_episodes, 'total_episodes');
        
        // Special styling for checkpoint approach
        if (data.approaching_checkpoint) {
            totalEpisodesElem.classList.add('checkpoint-approaching');
        } else {
            totalEpisodesElem.classList.remove('checkpoint-approaching');
        }
    }
    
    // Update reward stats
    if (data.avg_batch_reward !== undefined) {
        updateStatWithHighlight(
            avgBatchRewardElem, 
            +data.avg_batch_reward, 
            'avg_batch_reward', 
            val => val.toFixed(2)
        );
    }
    
    if (data.recent_avg_reward !== undefined) {
        updateStatWithHighlight(
            recentAvgRewardElem, 
            +data.recent_avg_reward, 
            'recent_avg_reward', 
            val => val.toFixed(2)
        );
    }
    
    if (data.best_avg_reward !== undefined) {
        updateStatWithHighlight(
            bestAvgRewardElem, 
            +data.best_avg_reward, 
            'best_avg_reward', 
            val => val.toFixed(2)
        );
    }
    
    // Update tile stats
    if (data.batch_max_tile !== undefined) {
        updateStatWithHighlight(
            batchMaxTileElem, 
            data.batch_max_tile, 
            'batch_max_tile'
        );
    }
    
    if (data.best_max_tile !== undefined) {
        updateStatWithHighlight(
            bestTileElem, 
            data.best_max_tile, 
            'best_max_tile'
        );
    }
    
    // Update moves
    if (data.avg_batch_moves !== undefined) {
        updateStatWithHighlight(
            avgEpisodeLengthElem, 
            +data.avg_batch_moves, 
            'avg_batch_moves', 
            val => val.toFixed(2)
        );
    }
    
    // Update learning rate
    if (data.current_lr !== undefined) {
        updateStatWithHighlight(
            currentLrElem, 
            +data.current_lr, 
            'current_lr', 
            val => val.toExponential(4)
        );
    }
    
    // Handle status message if provided
    if (data.status_message) {
        const checkpointInfo = document.getElementById('checkpoint-info');
        checkpointInfo.textContent = data.status_message;
        checkpointInfo.style.display = 'block';
        
        // Fade it out after 5 seconds
        setTimeout(() => {
            checkpointInfo.style.opacity = '0';
            setTimeout(() => {
                checkpointInfo.style.display = 'none';
                checkpointInfo.style.opacity = '1';
            }, 500);
        }, 5000);
    }
}

// Button event handlers
trainingButton.addEventListener('click', function() {
    // If we're already in training mode, do nothing
    if (currentMode === 'train') {
        return;
    }
    
    // If we're in watch mode, stop it first
    if (currentMode === 'watch') {
        // Stop current visualization
        socket.emit('stop');
        showLoadingIndicator(true, 'Switching to training mode...');
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Switch to training mode
            currentMode = 'train';
            socket.emit('start', { mode: 'train', hyperparams: hyperparams });
            
            trainingPanel.classList.remove('hidden');
            watchPanel.classList.add('hidden');
            
            // Update button states
            updateButtonStates();
            showLoadingIndicator(false);
        }, 1000);
        
        return;
    }
    
    // Standard flow for starting training from stopped state
    showLoadingIndicator(true, 'Starting training session...');
    currentMode = 'train';
    socket.emit('start', { mode: 'train', hyperparams: hyperparams });
    
    trainingPanel.classList.remove('hidden');
    watchPanel.classList.add('hidden');
    
    // Update button states
    updateButtonStates();
    
    // Hide loading after a brief delay
    setTimeout(() => showLoadingIndicator(false), 1500);
});

// Stop training button handler
stopTrainingButton.addEventListener('click', function() {
    if (currentMode !== 'train') {
        return;
    }
    
    showLoadingIndicator(true, 'Stopping training...');
    socket.emit('stop');
    
    // Wait for the server to confirm stop
    setTimeout(() => {
        currentMode = null;
        updateButtonStates();
        showLoadingIndicator(false);
        showToast('Training stopped', 'info');
    }, 1000);
});

watchButton.addEventListener('click', function() {
    // If we're already in watch mode, simply restart
    if (currentMode === 'watch') {
        // Stop current visualization
        socket.emit('stop');
        showLoadingIndicator(true, 'Restarting visualization...');
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Start a new visualization
            socket.emit('start', { mode: 'watch' });
            
            // Keep watch mode
            currentMode = 'watch';
            
            // Initialize new game board
            initGameBoard();
            showLoadingIndicator(false);
        }, 1000);
        
        return;
    }
    
    // If we're in training mode, stop it first
    if (currentMode === 'train') {
        // Stop current training
        socket.emit('stop');
        showLoadingIndicator(true, 'Switching to watch mode...');
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Switch to watch mode
            currentMode = 'watch';
            socket.emit('start', { mode: 'watch' });
            
            trainingPanel.classList.add('hidden');
            watchPanel.classList.remove('hidden');
            
            // Initialize new game board
            initGameBoard();
            
            // Update button states
            updateButtonStates();
            showLoadingIndicator(false);
        }, 1000);
        
        return;
    }
    
    // Standard flow for starting watch mode from stopped state
    showLoadingIndicator(true, 'Starting visualization...');
    currentMode = 'watch';
    socket.emit('start', { mode: 'watch' });
    
    trainingPanel.classList.add('hidden');
    watchPanel.classList.remove('hidden');
    
    initGameBoard();
    
    // Update button states
    updateButtonStates();
    
    // Hide loading after a brief delay
    setTimeout(() => showLoadingIndicator(false), 1500);
});

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('Connected to server');
    showToast('Connected to server', 'success');
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
    showToast('Disconnected from server', 'error');
});

// Handle mode changes from the server
socket.on('mode_change', function(data) {
    console.log('Mode changed to:', data.mode);
    currentMode = data.mode;
    
    // Update UI based on the new mode
    if (data.mode === 'train') {
        trainingPanel.classList.remove('hidden');
        watchPanel.classList.add('hidden');
    } else if (data.mode === 'watch') {
        trainingPanel.classList.add('hidden');
        watchPanel.classList.remove('hidden');
        // Initialize game board if we're switching to watch mode
        initGameBoard();
    } else if (data.mode === null) {
        // No active mode
        if (data.previous_mode === 'train') {
            // We were in training mode, keep showing training panel
            trainingPanel.classList.remove('hidden');
            watchPanel.classList.add('hidden');
        } else if (data.previous_mode === 'watch') {
            // We were in watch mode, keep showing game panel
            trainingPanel.classList.add('hidden');
            watchPanel.classList.remove('hidden');
        }
    }
    
    // Update button states
    updateButtonStates();
});

socket.on('hardware_info', function(data) {
    updateHardwareInfo(data);
});

// New optimized event handlers
socket.on('stats_update', function(data) {
    // Make sure we're in training mode
    currentMode = 'train';
    updateButtonStates();
    
    // Update UI immediately
    updateTrainingStats(data);
});

socket.on('chart_update', function(data) {
    // Update charts with the data
    updateCharts(data);
});

// Legacy support for older message format
socket.on('training_update', function(data) {
    // Make sure we're in training mode and update button states
    currentMode = 'train';
    updateButtonStates();
    
    // Update all stats with legacy data
    updateTrainingStats(data);
    
    // Check for chart data in legacy format
    if (data.rewards_chart && data.rewards_chart.length > 0) {
        updateCharts({
            rewards_chart: data.rewards_chart,
            max_tile_chart: data.max_tile_chart || [],
            loss_chart: data.loss_chart || [],
            moves_chart: data.moves_chart || [],
            episode_base: data.total_episodes - data.rewards_chart.length + 1
        });
    }
});

socket.on('game_update', function(data) {
    // Make sure we're in watch mode and update button states
    currentMode = 'watch';
    updateButtonStates();
    
    updateGameBoard(data.board);
    gameScore.textContent = data.score.toFixed(2);
    watchBestTile.textContent = data.max_tile;
    moveCount.textContent = data.moves;
});

socket.on('process_stopped', function() {
    // Reset the current mode
    currentMode = null;
    
    // Update all button states
    updateButtonStates();
    showToast('Process stopped', 'info');
});

socket.on('server_url', function(data) {
    document.getElementById('server-url').textContent = data.url;
});

socket.on('hyperparams_updated', function(data) {
    showToast('Hyperparameters updated successfully', 'success');
    
    // Optional: Update local hyperparams if server sent updates
    if (data && data.hyperparams) {
        Object.assign(hyperparams, data.hyperparams);
        updateHyperparamControls();
    }
});

// Update UI controls to match current hyperparams
function updateHyperparamControls() {
    for (const [paramName, controls] of Object.entries(hyperparamControls)) {
        const value = hyperparams[paramName];
        
        if (controls.slider && controls.valueInput) {
            controls.slider.value = value;
            controls.valueInput.value = value;
        } else if (controls.select) {
            controls.select.value = value;
        } else if (controls.toggle) {
            controls.toggle.checked = value;
        }
    }
}

// Add a loading indicator function
function showLoadingIndicator(show, message = "Processing...") {
    if (show) {
        // Create loading overlay if it doesn't exist
        if (!document.querySelector('.loading-overlay')) {
            const overlay = document.createElement('div');
            overlay.className = 'loading-overlay';
            
            const spinner = document.createElement('div');
            spinner.className = 'loading-spinner';
            
            const text = document.createElement('div');
            text.className = 'loading-text';
            text.textContent = message;
            
            overlay.appendChild(spinner);
            overlay.appendChild(text);
            document.body.appendChild(overlay);
        } else {
            // Update existing message
            document.querySelector('.loading-text').textContent = message;
            document.querySelector('.loading-overlay').style.display = 'flex';
        }
    } else {
        // Hide the loading overlay if it exists
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
}

// Toggle sections
function toggleSection(section, button) {
    const isOpen = section.style.maxHeight;
    
    if (isOpen) {
        section.style.maxHeight = null;
        button.querySelector('i').textContent = 'expand_more';
    } else {
        // Add some extra space for padding
        section.style.maxHeight = (section.scrollHeight + 20) + 'px';
        button.querySelector('i').textContent = 'expand_less';
    }
    
    // Force reflow to ensure transition works properly
    section.offsetHeight;
}

// Function to update button states based on current mode
function updateButtonStates() {
    if (currentMode === 'watch') {
        // Change Watch button to "Restart" with different color
        watchButton.innerHTML = '<i class="material-icons">refresh</i>Restart Game';
        watchButton.classList.add('btn-restart');
        watchButton.disabled = false;
        
        // Keep Training button enabled but with different visual style
        trainingButton.disabled = false;
        trainingButton.classList.add('btn-alternate');
        
        // Hide stop training button
        stopTrainingButton.classList.add('hidden');
    } else if (currentMode === 'train') {
        // Reset Watch button
        watchButton.innerHTML = '<i class="material-icons">visibility</i>Watch Gameplay';
        watchButton.classList.remove('btn-restart');
        watchButton.disabled = false;
        watchButton.classList.add('btn-alternate');
        
        // Hide Training button and show Stop button
        trainingButton.classList.add('hidden');
        stopTrainingButton.classList.remove('hidden');
    } else {
        // Reset all buttons to default state when stopped
        watchButton.innerHTML = '<i class="material-icons">visibility</i>Watch Gameplay';
        watchButton.classList.remove('btn-restart');
        watchButton.classList.remove('btn-alternate');
        watchButton.disabled = false;
        
        trainingButton.disabled = false;
        trainingButton.classList.remove('hidden');
        trainingButton.classList.remove('btn-alternate');
        
        // Hide stop training button
        stopTrainingButton.classList.add('hidden');
    }
}

// Initialize on page load
// Checkpoint management functions
function loadCheckpointInfo() {
    // Show loading state
    checkpointStatusLoading.classList.remove('hidden');
    checkpointStatusEmpty.classList.add('hidden');
    checkpointStatusError.classList.add('hidden');
    checkpointStatusLoaded.classList.add('hidden');
    
    // Fetch checkpoint info from server
    fetch('/checkpoint_info')
        .then(response => response.json())
        .then(data => {
            // Hide loading state
            checkpointStatusLoading.classList.add('hidden');
            
            if (!data.exists) {
                // Show empty state
                checkpointStatusEmpty.classList.remove('hidden');
                return;
            }
            
            // Update checkpoint info
            checkpointCreated.textContent = data.created;
            checkpointAge.textContent = data.age + ' ago';
            checkpointTrainingTime.textContent = data.training_time;
            checkpointEpisodes.textContent = data.episodes;
            checkpointReward.textContent = data.best_reward;
            checkpointBestTile.textContent = data.best_tile;
            checkpointSize.textContent = data.size;
            
            // Show loaded state
            checkpointStatusLoaded.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error fetching checkpoint info:', error);
            // Show error state
            checkpointStatusLoading.classList.add('hidden');
            checkpointErrorMessage.textContent = 'Error loading checkpoint information';
            checkpointStatusError.classList.remove('hidden');
        });
}

// Initialize and handle confirmation dialog
function showConfirmationDialog(title, message, confirmCallback) {
    // Set dialog content
    dialogTitle.textContent = title;
    dialogMessage.textContent = message;
    
    // Show dialog
    confirmationDialog.classList.remove('hidden');
    
    // Handle confirm button
    const handleConfirm = () => {
        confirmCallback();
        confirmationDialog.classList.add('hidden');
        dialogConfirm.removeEventListener('click', handleConfirm);
        dialogCancel.removeEventListener('click', handleCancel);
        dialogClose.removeEventListener('click', handleCancel);
    };
    
    // Handle cancel/close buttons
    const handleCancel = () => {
        confirmationDialog.classList.add('hidden');
        dialogConfirm.removeEventListener('click', handleConfirm);
        dialogCancel.removeEventListener('click', handleCancel);
        dialogClose.removeEventListener('click', handleCancel);
    };
    
    // Add event listeners
    dialogConfirm.addEventListener('click', handleConfirm);
    dialogCancel.addEventListener('click', handleCancel);
    dialogClose.addEventListener('click', handleCancel);
}

document.addEventListener('DOMContentLoaded', function() {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);
    themeSelector.value = savedTheme;
    
    // Theme selector event listener
    themeSelector.addEventListener('change', function() {
        setTheme(this.value);
    });
    
    // Toggle hardware panel
    toggleHardware.addEventListener('click', function() {
        toggleSection(hardwarePanel, this);
    });
    
    // Toggle hyperparameters panel
    toggleHyperparams.addEventListener('click', function() {
        toggleSection(hyperparamsContent, this);
    });
    
    // Toggle checkpoint panel and load checkpoint info when opened
    toggleCheckpoint.addEventListener('click', function() {
        toggleSection(checkpointPanel, this);
        
        // If panel is being opened, fetch checkpoint info
        if (checkpointPanel.style.maxHeight) {
            loadCheckpointInfo();
        }
    });
    
    // Setup tabs
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to current button and content
            this.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
    
    // Setup hyperparameter controls
    setupHyperparamControls();
    
    // Apply and reset hyperparameters buttons
    applyHyperparamsButton.addEventListener('click', applyHyperparameters);
    resetHyperparamsButton.addEventListener('click', resetHyperparameters);
    
    // Initialize charts
    initCharts();
    
    // Make sure buttons are in correct initial state
    updateButtonStates();
    
    // Open hardware panel by default
    toggleSection(hardwarePanel, toggleHardware);
    
    // Add delete checkpoint event listener
    deleteCheckpoint.addEventListener('click', function() {
        showConfirmationDialog(
            'Delete Checkpoint',
            'Are you sure you want to delete the current checkpoint? This action cannot be undone.',
            function() {
                // Send delete request to server
                fetch('/delete_checkpoint', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showToast('Checkpoint deleted successfully', 'success');
                        loadCheckpointInfo(); // Reload checkpoint info
                    } else {
                        showToast('Error: ' + data.message, 'error');
                    }
                })
                .catch(error => {
                    console.error('Error deleting checkpoint:', error);
                    showToast('Error deleting checkpoint', 'error');
                });
            }
        );
    });
});