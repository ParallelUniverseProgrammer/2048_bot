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

// State variables
let currentMode = null;
let charts = {};
let lastCheckpointTime = 0;

// Initialize Charts
function initCharts() {
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // Faster updates
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.7)'
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
            }
        }
    };

    // Create reward chart
    charts.reward = new Chart(rewardChart, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Batch Reward',
                data: [],
                borderColor: '#6200ea',
                backgroundColor: 'rgba(98, 0, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: 'Smooth Average',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Rewards Over Time',
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
                label: 'Max Tile Value',
                data: [],
                borderColor: '#f65e3b',
                backgroundColor: 'rgba(246, 94, 59, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Maximum Tile Over Time',
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
                label: 'Batch Loss',
                data: [],
                borderColor: '#cf6679',
                backgroundColor: 'rgba(207, 102, 121, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Training Loss',
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
                label: 'Episode Length',
                data: [],
                borderColor: '#18ffff',
                backgroundColor: 'rgba(24, 255, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: 'Smoothed Average',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.0)',
                borderWidth: 3,
                fill: false,
                tension: 0.6,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Moves Per Episode',
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

// Add data point to a chart
function addDataPoint(chart, label, ...dataPoints) {
    chart.data.labels.push(label);
    
    // If we have too many points, remove the oldest
    if (chart.data.labels.length > 100) {
        chart.data.labels.shift();
        chart.data.datasets.forEach(dataset => dataset.data.shift());
    }
    
    // Add new data points to each dataset
    chart.data.datasets.forEach((dataset, index) => {
        if (index < dataPoints.length) {
            dataset.data.push(dataPoints[index]);
        }
    });
    
    chart.update();
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
        }, 1000);
        
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
        }, 1000);
        
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
        }, 1000);
        
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

socket.on('training_update', function(data) {
    console.log("Received training update:", data);
    
    // Hide loading indicator on first training update
    showLoadingIndicator(false);
    
    // Make sure we're in training mode and update button states
    currentMode = 'train';
    updateButtonStates();
    
    // Check for checkpoint info
    if (data.status_message) {
        // Show checkpoint info message
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
    
    // Mark episode number with special style if approaching checkpoint
    const episodeCounter = document.getElementById('total-episodes');
    if (data.approaching_checkpoint) {
        episodeCounter.classList.add('checkpoint-approaching');
    } else {
        episodeCounter.classList.remove('checkpoint-approaching');
    }
    
    // Update training statistics display with formatted values
    // Make sure values are numeric before formatting them
    const avgBatchReward = typeof data.avg_batch_reward === 'number' ? data.avg_batch_reward : parseFloat(data.avg_batch_reward || 0);
    const recentAvgReward = typeof data.recent_avg_reward === 'number' ? data.recent_avg_reward : parseFloat(data.recent_avg_reward || 0);
    const bestAvgReward = typeof data.best_avg_reward === 'number' ? data.best_avg_reward : parseFloat(data.best_avg_reward || 0);
    const avgBatchMoves = typeof data.avg_batch_moves === 'number' ? data.avg_batch_moves : parseFloat(data.avg_batch_moves || 0);
    const batchMaxTile = typeof data.batch_max_tile === 'number' ? data.batch_max_tile : parseInt(data.batch_max_tile || 0);
    const bestMaxTile = typeof data.best_max_tile === 'number' ? data.best_max_tile : parseInt(data.best_max_tile || 0);
    const bestTileRate = typeof data.best_tile_rate === 'number' ? data.best_tile_rate : parseFloat(data.best_tile_rate || 0);
    const currentLr = typeof data.current_lr === 'number' ? data.current_lr : parseFloat(data.current_lr || 0);
    // Special parsing for total episodes - force numeric parsing
    let totalEpisodes = 0;
    try {
        if (typeof data.total_episodes === 'number') {
            totalEpisodes = data.total_episodes;
        } else if (typeof data.total_episodes === 'string') {
            totalEpisodes = parseInt(data.total_episodes.trim(), 10);
        } else if (data.total_episodes) {
            totalEpisodes = parseInt(String(data.total_episodes).trim(), 10);
        }
        // If parsing failed or resulted in NaN, use 0
        if (isNaN(totalEpisodes)) {
            console.error("Failed to parse total_episodes:", data.total_episodes);
            totalEpisodes = 0;
        }
    } catch (error) {
        console.error("Error parsing total_episodes:", error);
        totalEpisodes = 0;
    }
    
    console.log("Total episodes debug:", {
        original: data.total_episodes,
        parsed: totalEpisodes,
        type: typeof data.total_episodes
    });
    
    document.getElementById('avg-batch-reward').textContent = avgBatchReward.toFixed(2);
    document.getElementById('recent-avg-reward').textContent = recentAvgReward.toFixed(2);
    document.getElementById('best-avg-reward').textContent = bestAvgReward.toFixed(2);
    document.getElementById('avg-episode-length').textContent = avgBatchMoves.toFixed(2);
    document.getElementById('batch-max-tile').textContent = batchMaxTile;
    document.getElementById('best-tile').textContent = bestMaxTile;
    document.getElementById('best-tile-rate').textContent = bestTileRate.toFixed(1) + '%';
    document.getElementById('current-lr').textContent = currentLr.toExponential(4);
    document.getElementById('total-episodes').textContent = totalEpisodes;
    
    // Log the parsed values for debugging
    console.log("Parsed values:", {
        avgBatchReward, recentAvgReward, bestAvgReward, avgBatchMoves, 
        batchMaxTile, bestMaxTile, bestTileRate, currentLr, totalEpisodes
    });
    
    // Update charts with history arrays if available
    if (data.rewards_chart && data.rewards_chart.length > 0) {
        console.log(`Updating charts with ${data.rewards_chart.length} data points`);
        console.log("First few reward data points:", data.rewards_chart.slice(0, 5));
        console.log("First few max tile data points:", data.max_tile_chart ? data.max_tile_chart.slice(0, 5) : "No data");
        console.log("First few loss data points:", data.loss_chart ? data.loss_chart.slice(0, 5) : "No data");
        console.log("First few moves data points:", data.moves_chart ? data.moves_chart.slice(0, 5) : "No data");
        
        // Make sure we have valid arrays of numbers for all charts
        const rewardsData = Array.isArray(data.rewards_chart) ? 
            data.rewards_chart.map(val => typeof val === 'number' ? val : parseFloat(val || 0)) : [];
            
        const maxTileData = Array.isArray(data.max_tile_chart) ? 
            data.max_tile_chart.map(val => typeof val === 'number' ? val : parseInt(val || 0)) : [];
            
        const lossData = Array.isArray(data.loss_chart) ? 
            data.loss_chart.map(val => typeof val === 'number' ? val : parseFloat(val || 0)) : [];
            
        const movesData = Array.isArray(data.moves_chart) ? 
            data.moves_chart.map(val => typeof val === 'number' ? val : parseFloat(val || 0)) : [];
        
        // Log parsed data
        console.log("Parsed rewards data:", rewardsData.slice(0, 5));
        
        // Clear existing data
        charts.reward.data.labels = [];
        charts.reward.data.datasets[0].data = [];
        charts.reward.data.datasets[1].data = [];
        charts.maxTile.data.labels = [];
        charts.maxTile.data.datasets[0].data = [];
        charts.loss.data.labels = [];
        charts.loss.data.datasets[0].data = [];
        charts.moves.data.labels = [];
        charts.moves.data.datasets[0].data = [];
        
        // Add a second dataset for the smoothed line if it doesn't exist yet
        if (charts.moves.data.datasets.length < 2) {
            charts.moves.data.datasets.push({
                label: 'Smoothed Average',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.0)',
                borderWidth: 2,
                fill: false,
                tension: 0.6,
                pointRadius: 0,
                borderDash: [],
            });
        } else {
            charts.moves.data.datasets[1].data = [];
        }
        
        // Add all history points
        for (let i = 0; i < rewardsData.length; i++) {
            const episodeNum = totalEpisodes - rewardsData.length + i + 1;
            charts.reward.data.labels.push(episodeNum);
            charts.reward.data.datasets[0].data.push(rewardsData[i]);
            
            // Calculate a proper smoothed average for rewards like we do for moves
            if (i === 0) {
                // Clear the smoothed reward data first when starting new batch
                charts.reward.data.datasets[1].data = [];
            }
            
            // Apply moving average with 15-episode window (1.5x the original 10)
            const windowSize = 15;
            const startIdx = Math.max(0, i - windowSize + 1);
            const window = charts.reward.data.datasets[0].data.slice(startIdx, i + 1);
            const avgValue = window.reduce((sum, val) => sum + val, 0) / window.length;
            charts.reward.data.datasets[1].data.push(avgValue);
            
            // Max tile history
            charts.maxTile.data.labels.push(episodeNum);
            if (maxTileData.length > 0 && i < maxTileData.length) {
                charts.maxTile.data.datasets[0].data.push(maxTileData[i]);
            }
            
            // Loss history
            charts.loss.data.labels.push(episodeNum);
            if (lossData.length > 0 && i < lossData.length) {
                charts.loss.data.datasets[0].data.push(lossData[i]);
            }
            
            // Moves history (episode length)
            charts.moves.data.labels.push(episodeNum);
            if (movesData.length > 0 && i < movesData.length) {
                charts.moves.data.datasets[0].data.push(movesData[i]);
            } else if (avgBatchMoves) {
                // Fallback if moves_chart isn't available
                charts.moves.data.datasets[0].data.push(avgBatchMoves);
            }
            
            // Calculate smoothed average (10-episode moving average) for moves
            if (charts.moves.data.datasets[0].data.length > 0) {
                // Calculate the smoothed value for each point with a 10-episode window
                // This creates a proper smoothed line rather than just the latest value
                if (i === 0) {
                    // Clear the smoothed data first
                    charts.moves.data.datasets[1].data = [];
                }
                
                try {
                    const windowSize = 15;
                    const startIdx = Math.max(0, i - windowSize + 1);
                    const window = charts.moves.data.datasets[0].data.slice(startIdx, i + 1);
                    
                    // Make sure all values in the window are numbers
                    const numericWindow = window.map(v => typeof v === 'number' ? v : parseFloat(v || 0));
                    const avgValue = numericWindow.reduce((sum, val) => sum + val, 0) / numericWindow.length;
                    charts.moves.data.datasets[1].data.push(avgValue);
                } catch (error) {
                    console.error("Error calculating smoothed average:", error);
                    // Add a safe fallback value
                    charts.moves.data.datasets[1].data.push(charts.moves.data.datasets[0].data[i] || 0);
                }
            }
        }
        
        // Update all charts
        charts.reward.update();
        charts.maxTile.update();
        charts.loss.update();
        charts.moves.update();
    } else {
        // Fall back to single point update
        console.log("No history arrays found, using the optimized single point update");
        
        // Make sure both charts have two datasets
        if (charts.reward.data.datasets.length === 1) {
            charts.reward.data.datasets.push({
                label: 'Smoothed Average',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.0)',
                borderWidth: 3, 
                fill: false,
                tension: 0.6,
                pointRadius: 0
            });
        }
        
        // Add points to both datasets for the moves chart
        if (charts.moves.data.datasets.length === 1) {
            charts.moves.data.datasets.push({
                label: 'Smoothed Average',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.0)',
                borderWidth: 3,
                fill: false,
                tension: 0.6,
                pointRadius: 0
            });
        }
        
        // Calculate smoothed averages for both moves and rewards chart
        if (charts.moves.data.datasets[0].data.length > 0) {
            const windowSize = Math.min(15, charts.moves.data.datasets[0].data.length);
            
            // For moves chart
            const currentMovesData = [...charts.moves.data.datasets[0].data, avgBatchMoves];
            const smoothedMovesData = [];
            for (let i = 0; i < currentMovesData.length; i++) {
                try {
                    const startIdx = Math.max(0, i - windowSize + 1);
                    const window = currentMovesData.slice(startIdx, i + 1);
                    // Make sure all values in the window are numbers
                    const numericWindow = window.map(v => typeof v === 'number' ? v : parseFloat(v || 0));
                    const avgValue = numericWindow.reduce((sum, val) => sum + val, 0) / numericWindow.length;
                    smoothedMovesData.push(avgValue);
                } catch (error) {
                    console.error("Error calculating moves smoothed average:", error, "at index", i);
                    smoothedMovesData.push(currentMovesData[i] || 0);
                }
            }
            
            // For rewards chart
            const currentRewardsData = [...charts.reward.data.datasets[0].data, avgBatchReward];
            const smoothedRewardsData = [];
            for (let i = 0; i < currentRewardsData.length; i++) {
                try {
                    const startIdx = Math.max(0, i - windowSize + 1);
                    const window = currentRewardsData.slice(startIdx, i + 1);
                    // Make sure all values in the window are numbers
                    const numericWindow = window.map(v => typeof v === 'number' ? v : parseFloat(v || 0));
                    const avgValue = numericWindow.reduce((sum, val) => sum + val, 0) / numericWindow.length;
                    smoothedRewardsData.push(avgValue);
                } catch (error) {
                    console.error("Error calculating rewards smoothed average:", error, "at index", i);
                    smoothedRewardsData.push(currentRewardsData[i] || 0);
                }
            }
            
            // Update moves chart with the latest point and smoothed value
            const latestMovesSmoothed = smoothedMovesData[smoothedMovesData.length - 1];
            addDataPoint(charts.moves, totalEpisodes, avgBatchMoves, latestMovesSmoothed);
            
            // Update rewards chart with the latest point and smoothed value
            const latestRewardsSmoothed = smoothedRewardsData[smoothedRewardsData.length - 1];
            addDataPoint(charts.reward, totalEpisodes, avgBatchReward, latestRewardsSmoothed);
            
            // Update all previous points' smoothed values for moves
            for (let i = 0; i < charts.moves.data.datasets[1].data.length - 1; i++) {
                if (i < smoothedMovesData.length - 1) {
                    charts.moves.data.datasets[1].data[i] = smoothedMovesData[i];
                }
            }
            
            // Update all previous points' smoothed values for rewards
            for (let i = 0; i < charts.reward.data.datasets[1].data.length - 1; i++) {
                if (i < smoothedRewardsData.length - 1) {
                    charts.reward.data.datasets[1].data[i] = smoothedRewardsData[i];
                }
            }
            
            // Update max tile chart
            addDataPoint(charts.maxTile, totalEpisodes, batchMaxTile);
            
            // Update loss chart - make sure we have a valid value
            const lossValue = typeof data.batch_loss === 'number' ? data.batch_loss : parseFloat(data.batch_loss || 0);
            addDataPoint(charts.loss, totalEpisodes, lossValue);
        } else {
            // First data point - add to all charts
            addDataPoint(charts.moves, totalEpisodes, avgBatchMoves, avgBatchMoves);
            addDataPoint(charts.reward, totalEpisodes, avgBatchReward, avgBatchReward);
            addDataPoint(charts.maxTile, totalEpisodes, batchMaxTile);
            // Make sure we have a valid loss value
            const lossValue = typeof data.batch_loss === 'number' ? data.batch_loss : parseFloat(data.batch_loss || 0);
            addDataPoint(charts.loss, totalEpisodes, lossValue);
        }
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

// Add a loading indicator function - without notifications
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