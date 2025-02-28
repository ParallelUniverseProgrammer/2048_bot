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
    currentMode = 'train';
    socket.emit('start', { mode: 'train' });
    
    trainingPanel.classList.remove('hidden');
    watchPanel.classList.add('hidden');
    
    trainingButton.disabled = true;
    watchButton.disabled = true;
    stopButton.disabled = false;
});

watchButton.addEventListener('click', function() {
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
    
    trainingButton.disabled = false;
    watchButton.disabled = false;
    stopButton.disabled = true;
    
    currentMode = null;
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
    
    // Update training statistics display
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
        
        // Add all history points
        for (let i = 0; i < data.rewards_chart.length; i++) {
            const episodeNum = data.total_episodes - data.rewards_chart.length + i + 1;
            charts.reward.data.labels.push(episodeNum);
            charts.reward.data.datasets[0].data.push(data.rewards_chart[i]);
            
            // Recent average data - just use the same value for now
            charts.reward.data.datasets[1].data.push(data.recent_avg_reward);
            
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
        }
        
        // Update all charts
        charts.reward.update();
        charts.maxTile.update();
        charts.loss.update();
        charts.moves.update();
    } else {
        // Fall back to single point update
        console.log("No history arrays found, adding single data points");
        addDataPoint(charts.reward, data.total_episodes, data.avg_batch_reward, data.recent_avg_reward);
        addDataPoint(charts.maxTile, data.total_episodes, data.batch_max_tile);
        addDataPoint(charts.loss, data.total_episodes, data.batch_loss);
        addDataPoint(charts.moves, data.total_episodes, data.avg_batch_moves);
    }
});

socket.on('game_update', function(data) {
    updateGameBoard(data.board);
    gameScore.textContent = data.score.toFixed(2);
    bestTile.textContent = data.max_tile;
    moveCount.textContent = data.moves;
});

socket.on('process_stopped', function() {
    trainingButton.disabled = false;
    watchButton.disabled = false;
    stopButton.disabled = true;
    currentMode = null;
});

socket.on('server_url', function(data) {
    document.getElementById('server-url').textContent = data.url;
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    stopButton.disabled = true;
});