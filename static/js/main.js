// Initialize Socket.IO
const socket = io();

// DOM Elements
const trainingButton = document.getElementById('start-training');
const watchButton = document.getElementById('start-watch');
const stopButton = document.getElementById('stop-process');
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
                label: 'Recent Avg Reward',
                data: [],
                borderColor: '#03dac6',
                backgroundColor: 'rgba(3, 218, 198, 0.1)',
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
    // Don't allow if a process is currently stopping
    if (stopButton.textContent === "Stopping...") {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.style.backgroundColor = "#cf6679"; // Error color
        notification.textContent = 'Please wait for the current process to stop completely';
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 500);
        }, 3000);
        return;
    }
    
    // Show loading indicator until we get the first update
    showLoadingIndicator(true, "Starting training process...");
    
    currentMode = 'train';
    socket.emit('start', { mode: 'train' });
    
    trainingPanel.classList.remove('hidden');
    watchPanel.classList.add('hidden');
    
    trainingButton.disabled = true;
    watchButton.disabled = true;
    stopButton.disabled = false;
});

watchButton.addEventListener('click', function() {
    // Don't allow if a process is currently stopping
    if (stopButton.textContent === "Stopping...") {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.style.backgroundColor = "#cf6679"; // Error color
        notification.textContent = 'Please wait for the current process to stop completely';
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 500);
        }, 3000);
        return;
    }
    
    // Show loading indicator until we get the first game update
    showLoadingIndicator(true, "Loading game visualization...");
    
    currentMode = 'watch';
    socket.emit('start', { mode: 'watch' });
    
    trainingPanel.classList.add('hidden');
    watchPanel.classList.remove('hidden');
    
    trainingButton.disabled = true;
    watchButton.disabled = true;
    stopButton.disabled = false;
    
    initGameBoard();
});

stopButton.addEventListener('click', function() {
    socket.emit('stop');
    
    // Don't immediately re-enable buttons
    // Wait for the stopping_process and process_stopped events
    stopButton.textContent = "Stopping...";
    stopButton.style.backgroundColor = "#f65e3b";
    stopButton.disabled = true;
    
    // Disable both buttons during stopping
    trainingButton.disabled = true;
    watchButton.disabled = true;
    watchButton.style.opacity = "0.5";
});

// Socket.IO event handlers
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('disconnect', function() {
    console.log('Disconnected from server');
});

socket.on('stopping_process', function() {
    // Visual feedback that stopping is in progress
    stopButton.textContent = "Stopping...";
    stopButton.style.backgroundColor = "#f65e3b";
    stopButton.disabled = true;
    
    // Disable watch button while stopping is in progress
    watchButton.disabled = true;
    watchButton.style.opacity = "0.5";
});

socket.on('hardware_info', function(data) {
    updateHardwareInfo(data);
});

socket.on('training_update', function(data) {
    console.log("Received training update:", data);
    
    // Hide loading indicator on first training update
    showLoadingIndicator(false);
    
    // Update training statistics display with formatted values
    document.getElementById('avg-batch-reward').textContent = data.avg_batch_reward.toFixed(2);
    document.getElementById('recent-avg-reward').textContent = data.recent_avg_reward.toFixed(2);
    document.getElementById('best-avg-reward').textContent = data.best_avg_reward.toFixed(2);
    document.getElementById('avg-episode-length').textContent = data.avg_batch_moves.toFixed(2);
    document.getElementById('batch-max-tile').textContent = data.batch_max_tile;
    document.getElementById('best-tile').textContent = data.best_max_tile;
    document.getElementById('best-tile-rate').textContent = data.best_tile_rate.toFixed(1) + '%';
    document.getElementById('current-lr').textContent = data.current_lr.toExponential(4);
    document.getElementById('total-episodes').textContent = data.total_episodes;
    
    // Update charts with history arrays if available
    if (data.rewards_chart && data.rewards_chart.length > 0) {
        console.log(`Updating charts with ${data.rewards_chart.length} data points`);
        
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
        for (let i = 0; i < data.rewards_chart.length; i++) {
            const episodeNum = data.total_episodes - data.rewards_chart.length + i + 1;
            charts.reward.data.labels.push(episodeNum);
            charts.reward.data.datasets[0].data.push(data.rewards_chart[i]);
            
            // Calculate a proper smoothed average for rewards like we do for moves
            if (i === 0) {
                // Clear the smoothed reward data first when starting new batch
                charts.reward.data.datasets[1].data = [];
            }
            
            // Apply moving average with 10-episode window
            const windowSize = 10;
            const startIdx = Math.max(0, i - windowSize + 1);
            const window = charts.reward.data.datasets[0].data.slice(startIdx, i + 1);
            const avgValue = window.reduce((sum, val) => sum + val, 0) / window.length;
            charts.reward.data.datasets[1].data.push(avgValue);
            
            // Max tile history
            charts.maxTile.data.labels.push(episodeNum);
            if (data.max_tile_chart && i < data.max_tile_chart.length) {
                charts.maxTile.data.datasets[0].data.push(data.max_tile_chart[i]);
            }
            
            // Loss history
            charts.loss.data.labels.push(episodeNum);
            if (data.loss_chart && i < data.loss_chart.length) {
                charts.loss.data.datasets[0].data.push(data.loss_chart[i]);
            }
            
            // Moves history (episode length)
            charts.moves.data.labels.push(episodeNum);
            if (data.moves_chart && i < data.moves_chart.length) {
                charts.moves.data.datasets[0].data.push(data.moves_chart[i]);
            } else if (data.avg_batch_moves) {
                // Fallback if moves_chart isn't available
                charts.moves.data.datasets[0].data.push(data.avg_batch_moves);
            }
            
            // Calculate smoothed average (10-episode moving average) for moves
            if (charts.moves.data.datasets[0].data.length > 0) {
                // Calculate the smoothed value for each point with a 10-episode window
                // This creates a proper smoothed line rather than just the latest value
                if (i === 0) {
                    // Clear the smoothed data first
                    charts.moves.data.datasets[1].data = [];
                }
                
                const windowSize = 10;
                const startIdx = Math.max(0, i - windowSize + 1);
                const window = charts.moves.data.datasets[0].data.slice(startIdx, i + 1);
                const avgValue = window.reduce((sum, val) => sum + val, 0) / window.length;
                charts.moves.data.datasets[1].data.push(avgValue);
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
            const windowSize = Math.min(10, charts.moves.data.datasets[0].data.length);
            
            // For moves chart
            const currentMovesData = [...charts.moves.data.datasets[0].data, data.avg_batch_moves];
            const smoothedMovesData = [];
            for (let i = 0; i < currentMovesData.length; i++) {
                const startIdx = Math.max(0, i - windowSize + 1);
                const window = currentMovesData.slice(startIdx, i + 1);
                const avgValue = window.reduce((sum, val) => sum + val, 0) / window.length;
                smoothedMovesData.push(avgValue);
            }
            
            // For rewards chart
            const currentRewardsData = [...charts.reward.data.datasets[0].data, data.avg_batch_reward];
            const smoothedRewardsData = [];
            for (let i = 0; i < currentRewardsData.length; i++) {
                const startIdx = Math.max(0, i - windowSize + 1);
                const window = currentRewardsData.slice(startIdx, i + 1);
                const avgValue = window.reduce((sum, val) => sum + val, 0) / window.length;
                smoothedRewardsData.push(avgValue);
            }
            
            // Update moves chart with the latest point and smoothed value
            const latestMovesSmoothed = smoothedMovesData[smoothedMovesData.length - 1];
            addDataPoint(charts.moves, data.total_episodes, data.avg_batch_moves, latestMovesSmoothed);
            
            // Update rewards chart with the latest point and smoothed value
            const latestRewardsSmoothed = smoothedRewardsData[smoothedRewardsData.length - 1];
            addDataPoint(charts.reward, data.total_episodes, data.avg_batch_reward, latestRewardsSmoothed);
            
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
            addDataPoint(charts.maxTile, data.total_episodes, data.batch_max_tile);
            
            // Update loss chart 
            addDataPoint(charts.loss, data.total_episodes, data.batch_loss);
        } else {
            // First data point - add to all charts
            addDataPoint(charts.moves, data.total_episodes, data.avg_batch_moves, data.avg_batch_moves);
            addDataPoint(charts.reward, data.total_episodes, data.avg_batch_reward, data.avg_batch_reward);
            addDataPoint(charts.maxTile, data.total_episodes, data.batch_max_tile);
            addDataPoint(charts.loss, data.total_episodes, data.batch_loss);
        }
    }
});

socket.on('game_update', function(data) {
    // Hide loading indicator on first game update
    showLoadingIndicator(false);
    
    updateGameBoard(data.board);
    gameScore.textContent = data.score.toFixed(2);
    bestTile.textContent = data.max_tile;
    moveCount.textContent = data.moves;
});

socket.on('process_stopped', function() {
    // Re-enable buttons and reset their appearance
    trainingButton.disabled = false;
    watchButton.disabled = false;
    watchButton.style.opacity = "1";
    
    stopButton.disabled = true;
    stopButton.textContent = "Stop Process";
    stopButton.style.backgroundColor = ""; // Reset to default color
    
    currentMode = null;
    
    // Notify the user that the process has stopped
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = 'Process stopped successfully';
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 500);
    }, 3000);
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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    stopButton.disabled = true;
});