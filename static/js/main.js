// Initialize Socket.IO
const socket = io();

// DOM Elements
const trainingButton = document.getElementById('start-training');
const watchButton = document.getElementById('start-watch');
const trainingPanel = document.getElementById('training-panel');
const watchPanel = document.getElementById('watch-panel');
const gameBoard = document.getElementById('game-board');
const gameScore = document.getElementById('game-score');
const bestTile = document.getElementById('best-tile');
const moveCount = document.getElementById('move-count');

// Chart containers
const rewardChart = document.getElementById('reward-chart').getContext('2d');
const maxTileChart = document.getElementById('max-tile-chart').getContext('2d');
const lossChart = document.getElementById('loss-chart').getContext('2d');
const movesChart = document.getElementById('moves-chart').getContext('2d');

// Hardware monitoring elements
const cpuUsage = document.getElementById('cpu-usage');
const ramUsage = document.getElementById('ram-usage');
const gpuInfo = document.getElementById('gpu-info');

// Training stats elements (for direct updating)
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

// Initialize Charts
function initCharts() {
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
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.7)',
                    maxTicksLimit: 5 // Limit ticks for better performance
                },
                title: {
                    display: true,
                    text: 'Episode',
                    color: 'rgba(255, 255, 255, 0.7)'
                }
            },
            y: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.7)'
                }
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: 'rgba(255, 255, 255, 0.7)'
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
                borderColor: '#6200ea',
                backgroundColor: 'rgba(98, 0, 234, 0.1)',
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
                    color: 'rgba(255, 255, 255, 0.87)'
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
                    color: 'rgba(255, 255, 255, 0.87)'
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
                    color: 'rgba(255, 255, 255, 0.87)'
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
                borderColor: '#18ffff',
                backgroundColor: 'rgba(24, 255, 255, 0.1)',
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
                    color: 'rgba(255, 255, 255, 0.87)'
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
                    <div class="hardware-title">GPU ${index}: ${gpu.name}</div>
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
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Show loading indicator for training
            showLoadingIndicator(true, "Starting training process...");
            
            // Switch to training mode
            currentMode = 'train';
            socket.emit('start', { mode: 'train' });
            
            trainingPanel.classList.remove('hidden');
            watchPanel.classList.add('hidden');
            
            // Keep buttons disabled as process is running
            trainingButton.disabled = true;
            watchButton.disabled = true;
            
            // Update button states
            updateButtonStates();
        }, 500);
        
        return;
    }
    
    // Standard flow for starting training from stopped state
    showLoadingIndicator(true, "Starting training process...");
    
    currentMode = 'train';
    socket.emit('start', { mode: 'train' });
    
    trainingPanel.classList.remove('hidden');
    watchPanel.classList.add('hidden');
    
    trainingButton.disabled = true;
    watchButton.disabled = true;
    
    // Update button states
    updateButtonStates();
});

watchButton.addEventListener('click', function() {
    // If we're already in watch mode, simply restart
    if (currentMode === 'watch') {
        // Stop current visualization
        socket.emit('stop');
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Show loading indicator for restart
            showLoadingIndicator(true, "Restarting game visualization...");
            
            // Start a new visualization
            socket.emit('start', { mode: 'watch' });
            
            // Keep watch mode
            currentMode = 'watch';
            
            // Initialize new game board
            initGameBoard();
        }, 500);
        
        return;
    }
    
    // If we're in training mode, stop it first
    if (currentMode === 'train') {
        // Stop current training
        socket.emit('stop');
        
        // Wait briefly for the process to stop completely
        setTimeout(() => {
            // Show loading indicator for game visualization
            showLoadingIndicator(true, "Loading game visualization...");
            
            // Switch to watch mode
            currentMode = 'watch';
            socket.emit('start', { mode: 'watch' });
            
            trainingPanel.classList.add('hidden');
            watchPanel.classList.remove('hidden');
            
            // Keep buttons disabled as process is running
            trainingButton.disabled = true;
            watchButton.disabled = true;
            
            // Initialize new game board
            initGameBoard();
            
            // Update button states
            updateButtonStates();
        }, 500);
        
        return;
    }
    
    // Standard flow for starting watch mode from stopped state
    showLoadingIndicator(true, "Loading game visualization...");
    
    currentMode = 'watch';
    socket.emit('start', { mode: 'watch' });
    
    trainingPanel.classList.add('hidden');
    watchPanel.classList.remove('hidden');
    
    trainingButton.disabled = true;
    watchButton.disabled = true;
    
    initGameBoard();
    
    // Update button states
    updateButtonStates();
});

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
});

socket.on('hardware_info', function(data) {
    updateHardwareInfo(data);
});

// New optimized event handlers
socket.on('stats_update', function(data) {
    // Stop loading indicator
    showLoadingIndicator(false);
    
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
    // Hide loading indicator on first training update
    showLoadingIndicator(false);
    
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
    // Hide loading indicator on first game update
    showLoadingIndicator(false);
    
    // Make sure we're in watch mode and update button states
    currentMode = 'watch';
    updateButtonStates();
    
    updateGameBoard(data.board);
    gameScore.textContent = data.score.toFixed(2);
    bestTile.textContent = data.max_tile;
    moveCount.textContent = data.moves;
});

socket.on('process_stopped', function() {
    // Reset the current mode
    currentMode = null;
    
    // Update all button states
    updateButtonStates();
});

socket.on('server_url', function(data) {
    document.getElementById('server-url').textContent = data.url;
});

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

// Function to update button states based on current mode
function updateButtonStates() {
    if (currentMode === 'watch') {
        // Change Watch button to "Restart" with different color
        watchButton.textContent = "Restart Game";
        watchButton.classList.add('btn-restart');
        watchButton.disabled = false;
        
        // Keep Training button enabled but with different visual style
        trainingButton.disabled = false;
        trainingButton.classList.add('btn-alternate');
    } else if (currentMode === 'train') {
        // Reset Watch button
        watchButton.textContent = "Watch Gameplay";
        watchButton.classList.remove('btn-restart');
        watchButton.disabled = false;
        watchButton.classList.add('btn-alternate');
        
        // Disable Training button
        trainingButton.disabled = true;
        trainingButton.classList.remove('btn-alternate');
    } else {
        // Reset all buttons to default state when stopped
        watchButton.textContent = "Watch Gameplay";
        watchButton.classList.remove('btn-restart');
        watchButton.classList.remove('btn-alternate');
        watchButton.disabled = false;
        
        trainingButton.disabled = false;
        trainingButton.classList.remove('btn-alternate');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    // Make sure buttons are in correct initial state
    updateButtonStates();
});