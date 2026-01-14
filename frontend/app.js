/**
 * Quantum Elasticity Solver - Frontend Application
 * Handles API communication, UI updates, and chart rendering
 */

// ============================================
// Configuration
// ============================================
// For local dev: use localhost:8000
// For Koyeb/production: API is proxied on same origin
const API_BASE_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : '';

// Method color palette
const METHOD_COLORS = {
    'HHL (Spectral)': '#6366f1',
    'VQLS (Variational)': '#ec4899',
    'Adiabatic (Subaşı)': '#f59e0b',
    'Classical (Direct)': '#10b981'
};

const METHOD_CLASSES = {
    'HHL (Spectral)': 'hhl',
    'VQLS (Variational)': 'vqls',
    'Adiabatic (Subaşı)': 'adiabatic',
    'Classical (Direct)': 'classical'
};

// ============================================
// State
// ============================================
let convergenceChart = null;
let currentResults = null;
let chartScale = 'log';
let customMatrixEnabled = false;
let customMatrix = null;
let customVector = null;

// ============================================
// DOM Elements
// ============================================
const elements = {
    matrixSize: document.getElementById('matrix-size'),
    matrixSizeValue: document.getElementById('matrix-size-value'),
    forcePosition: document.getElementById('force-position'),
    methodHHL: document.getElementById('method-hhl'),
    methodVQLS: document.getElementById('method-vqls'),
    methodAdiabatic: document.getElementById('method-adiabatic'),
    methodClassical: document.getElementById('method-classical'),
    vqlsIter: document.getElementById('vqls-iter'),
    adiabaticSteps: document.getElementById('adiabatic-steps'),
    adiabaticTime: document.getElementById('adiabatic-time'),
    useCustomMatrix: document.getElementById('use-custom-matrix'),
    resetMatrixBtn: document.getElementById('reset-matrix-btn'),
    matrixModeBadge: document.getElementById('matrix-mode-badge'),
    solveBtn: document.getElementById('solve-btn'),
    matrixDisplay: document.getElementById('matrix-display'),
    vectorDisplay: document.getElementById('vector-display'),
    loadingContainer: document.getElementById('loading-container'),
    resultsContent: document.getElementById('results-content'),
    resultsStatus: document.getElementById('results-status'),
    methodCards: document.getElementById('method-cards'),
    solutionTable: document.getElementById('solution-table'),
    metricsGrid: document.getElementById('metrics-grid'),
    chartBtns: document.querySelectorAll('.chart-btn')
};

// ============================================
// Matrix Generation
// ============================================
function generateStiffnessMatrix(n) {
    const matrix = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < n; j++) {
            if (i === j) {
                row.push(i < n - 1 ? 2 : 1);
            } else if (Math.abs(i - j) === 1) {
                row.push(-1);
            } else {
                row.push(0);
            }
        }
        matrix.push(row);
    }
    return matrix;
}

function generateForceVector(n, position) {
    const vector = new Array(n).fill(0);
    const pos = position < 0 ? n - 1 : position;
    vector[Math.min(pos, n - 1)] = 1;
    return vector;
}

// ============================================
// UI Rendering
// ============================================
function renderMatrix(matrix, editable = false) {
    elements.matrixDisplay.innerHTML = '';
    matrix.forEach((row, i) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'matrix-row';
        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            if (i === j) cell.classList.add('diagonal');
            else if (val !== 0) cell.classList.add('non-zero');
            else cell.classList.add('zero');

            if (editable) {
                cell.classList.add('editable');
                const input = document.createElement('input');
                input.type = 'number';
                input.className = 'matrix-input';
                input.value = val;
                input.step = '0.1';
                input.dataset.row = i;
                input.dataset.col = j;
                input.addEventListener('change', handleMatrixChange);
                cell.appendChild(input);
            } else {
                cell.textContent = val;
            }
            rowDiv.appendChild(cell);
        });
        elements.matrixDisplay.appendChild(rowDiv);
    });
}

function renderVector(vector, editable = false) {
    elements.vectorDisplay.innerHTML = '';
    vector.forEach((val, i) => {
        const cell = document.createElement('div');
        cell.className = 'vector-cell';
        if (val !== 0) cell.classList.add('active');

        if (editable) {
            cell.classList.add('editable');
            const input = document.createElement('input');
            input.type = 'number';
            input.className = 'vector-input';
            input.value = val;
            input.step = '0.1';
            input.dataset.index = i;
            input.addEventListener('change', handleVectorChange);
            cell.appendChild(input);
        } else {
            cell.textContent = val;
        }
        elements.vectorDisplay.appendChild(cell);
    });
}

function handleMatrixChange(e) {
    const row = parseInt(e.target.dataset.row);
    const col = parseInt(e.target.dataset.col);
    const val = parseFloat(e.target.value) || 0;

    if (customMatrix) {
        customMatrix[row][col] = val;
    }
}

function handleVectorChange(e) {
    const idx = parseInt(e.target.dataset.index);
    const val = parseFloat(e.target.value) || 0;

    if (customVector) {
        customVector[idx] = val;
    }
}

function updateForceOptions(n) {
    const select = elements.forcePosition;
    const currentValue = select.value;

    // Keep the default option and rebuild numeric options
    while (select.options.length > 1) {
        select.remove(1);
    }

    for (let i = 0; i < n; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Node ${i}`;
        select.appendChild(option);
    }

    // Restore selection if valid
    if (parseInt(currentValue) < n) {
        select.value = currentValue;
    }
}

function updateVisualization() {
    const n = parseInt(elements.matrixSize.value);
    const forcePos = parseInt(elements.forcePosition.value);

    if (customMatrixEnabled) {
        // Initialize custom matrix/vector if size changed
        if (!customMatrix || customMatrix.length !== n) {
            customMatrix = generateStiffnessMatrix(n);
            customVector = generateForceVector(n, forcePos);
        }
        renderMatrix(customMatrix, true);
        renderVector(customVector, true);

        // Update UI indicators
        if (elements.matrixModeBadge) {
            elements.matrixModeBadge.textContent = 'Editable';
            elements.matrixModeBadge.classList.add('editable');
        }
        if (elements.resetMatrixBtn) {
            elements.resetMatrixBtn.style.display = 'block';
        }
    } else {
        const matrix = generateStiffnessMatrix(n);
        const vector = generateForceVector(n, forcePos);
        renderMatrix(matrix, false);
        renderVector(vector, false);

        // Update UI indicators
        if (elements.matrixModeBadge) {
            elements.matrixModeBadge.textContent = 'Visualization';
            elements.matrixModeBadge.classList.remove('editable');
        }
        if (elements.resetMatrixBtn) {
            elements.resetMatrixBtn.style.display = 'none';
        }
    }

    updateForceOptions(n);
}

function resetToDefault() {
    const n = parseInt(elements.matrixSize.value);
    const forcePos = parseInt(elements.forcePosition.value);
    customMatrix = generateStiffnessMatrix(n);
    customVector = generateForceVector(n, forcePos);
    updateVisualization();
}

function toggleCustomMatrix(enabled) {
    customMatrixEnabled = enabled;
    if (enabled) {
        const n = parseInt(elements.matrixSize.value);
        const forcePos = parseInt(elements.forcePosition.value);
        customMatrix = generateStiffnessMatrix(n);
        customVector = generateForceVector(n, forcePos);
    }
    updateVisualization();
}

// ============================================
// API Communication
// ============================================
async function solveProblem() {
    // Get configuration
    const methods = [];
    if (elements.methodHHL.checked) methods.push('hhl');
    if (elements.methodVQLS.checked) methods.push('vqls');
    if (elements.methodAdiabatic.checked) methods.push('adiabatic');
    if (elements.methodClassical.checked) methods.push('classical');

    if (methods.length === 0) {
        alert('Please select at least one method.');
        return;
    }

    const problem = {
        matrix_size: parseInt(elements.matrixSize.value),
        force_position: parseInt(elements.forcePosition.value),
        force_value: 1.0
    };

    // Add custom matrix/vector if enabled
    if (customMatrixEnabled && customMatrix && customVector) {
        problem.custom_matrix = customMatrix;
        problem.custom_force = customVector;
    }

    const config = {
        methods: methods,
        vqls_max_iter: parseInt(elements.vqlsIter.value),
        adiabatic_steps: parseInt(elements.adiabaticSteps.value),
        adiabatic_time: parseFloat(elements.adiabaticTime?.value || 300000)
    };

    // Show loading state
    setLoadingState(true);

    try {
        const response = await fetch(`${API_BASE_URL}/solve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ...problem, ...config })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        currentResults = data;
        renderResults(data);

    } catch (error) {
        console.error('Error solving problem:', error);
        showError(error.message);
    } finally {
        setLoadingState(false);
    }
}

function setLoadingState(loading) {
    elements.solveBtn.classList.toggle('loading', loading);
    elements.solveBtn.querySelector('.btn-text').textContent = loading ? 'Computing...' : 'Run Solver';
    elements.loadingContainer.style.display = loading ? 'flex' : 'none';
    elements.resultsContent.style.display = loading ? 'none' : (currentResults ? 'block' : 'none');

    // Update status
    const statusDot = elements.resultsStatus.querySelector('.status-dot');
    const statusText = elements.resultsStatus.querySelector('.status-text');

    if (loading) {
        statusDot.classList.add('active');
        statusText.textContent = 'Computing...';
    } else if (currentResults) {
        statusDot.classList.add('active');
        statusText.textContent = 'Results ready';
    } else {
        statusDot.classList.remove('active');
        statusText.textContent = 'Ready to compute';
    }
}

function showError(message) {
    const statusText = elements.resultsStatus.querySelector('.status-text');
    statusText.textContent = `Error: ${message}`;
    elements.resultsContent.style.display = 'none';
}

// ============================================
// Results Rendering
// ============================================
function renderResults(data) {
    elements.resultsContent.style.display = 'block';

    renderConvergenceChart(data.results);
    renderMethodCards(data.results);
    renderSolutionTable(data);
    renderMetrics(data);
}

function renderConvergenceChart(results) {
    const ctx = document.getElementById('convergence-chart').getContext('2d');

    // Destroy existing chart
    if (convergenceChart) {
        convergenceChart.destroy();
    }

    // Find the maximum actual iteration count across all methods
    let maxIterations = 100;
    results.forEach(result => {
        // Use cost_history_x if available, otherwise use iterations
        const maxX = result.cost_history_x
            ? Math.max(...result.cost_history_x)
            : result.iterations;
        if (maxX > maxIterations) {
            maxIterations = maxX;
        }
    });

    // Prepare datasets with actual iteration values on x-axis
    const datasets = results.map(result => {
        const xValues = result.cost_history_x || result.cost_history.map((_, i) => i);
        const yValues = result.cost_history;

        // Convert to {x, y} format using actual iteration numbers
        const formattedData = yValues.map((val, idx) => ({
            x: xValues[idx] || idx,
            y: Math.max(val, 1e-16) // Ensure positive values for log scale
        }));

        return {
            label: result.name,
            data: formattedData,
            borderColor: METHOD_COLORS[result.name] || '#888',
            backgroundColor: METHOD_COLORS[result.name] ? METHOD_COLORS[result.name] + '20' : 'transparent',
            borderWidth: 2.5,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2,
            fill: false
        };
    });

    // Add exact solution line spanning the full range
    datasets.push({
        label: 'Exact (QLSP)',
        data: [
            { x: 0, y: 1e-12 },
            { x: maxIterations, y: 1e-12 }
        ],
        borderColor: '#06b6d4',
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        fill: false
    });

    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 800,
                easing: 'easeOutQuart'
            },
            interaction: {
                intersect: false,
                mode: 'nearest'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        boxWidth: 12,
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 36, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        title: function (context) {
                            return `Iteration: ${Math.round(context[0].parsed.x)}`;
                        },
                        label: function (context) {
                            const yVal = context.parsed.y;
                            return `${context.dataset.label}: ${yVal.toExponential(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Iterations / Steps',
                        color: '#64748b',
                        font: { family: 'Inter', size: 12, weight: '500' }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#64748b',
                        maxTicksLimit: 10,
                        callback: function (value) {
                            // Format large numbers (e.g., 10000 -> 10k)
                            if (value >= 1000) {
                                return (value / 1000) + 'k';
                            }
                            return value;
                        }
                    },
                    min: 0,
                    max: maxIterations
                },
                y: {
                    type: chartScale === 'log' ? 'logarithmic' : 'linear',
                    title: {
                        display: true,
                        text: 'QLSP Error',
                        color: '#64748b',
                        font: { family: 'Inter', size: 12, weight: '500' }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#64748b',
                        callback: function (value) {
                            if (chartScale === 'log') {
                                // Show only powers of 10
                                const log = Math.log10(value);
                                if (Math.floor(log) === log) {
                                    return '10^' + log;
                                }
                                return '';
                            }
                            return value.toExponential(0);
                        }
                    },
                    min: chartScale === 'log' ? 1e-16 : 0,
                    max: 1
                }
            }
        }
    });
}

function extendCostHistory(history, targetLength) {
    const extended = [];
    const factor = Math.ceil(targetLength / history.length);

    for (let i = 0; i < targetLength; i++) {
        const idx = Math.min(Math.floor(i / factor), history.length - 1);
        extended.push(history[idx]);
    }

    return extended;
}

function renderMethodCards(results) {
    elements.methodCards.innerHTML = '';

    results.forEach(result => {
        const cardClass = METHOD_CLASSES[result.name] || 'classical';
        const typeLabel = result.name.includes('Classical') ? 'Classical' : 'Quantum';

        const card = document.createElement('div');
        card.className = `method-card ${cardClass}`;
        card.innerHTML = `
            <div class="method-card-header">
                <h4 class="method-card-title">${result.name}</h4>
                <span class="method-card-type">${typeLabel}</span>
            </div>
            <div class="method-card-stats">
                <div class="stat-row">
                    <span class="stat-label">Final Error</span>
                    <span class="stat-value ${result.final_cost < 1e-4 ? 'highlight' : 'error'}">${result.final_cost.toExponential(2)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Iterations</span>
                    <span class="stat-value">${result.iterations}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Time</span>
                    <span class="stat-value">${(result.execution_time * 1000).toFixed(1)} ms</span>
                </div>
            </div>
        `;

        elements.methodCards.appendChild(card);
    });
}

function renderSolutionTable(data) {
    const thead = elements.solutionTable.querySelector('thead tr');
    const tbody = elements.solutionTable.querySelector('tbody');

    // Clear existing
    thead.innerHTML = '<th>Node</th>';
    tbody.innerHTML = '';

    // Add headers
    data.results.forEach(result => {
        const th = document.createElement('th');
        th.textContent = result.name.split(' ')[0];
        th.style.color = METHOD_COLORS[result.name] || '#fff';
        thead.appendChild(th);
    });

    // Add rows
    const n = data.problem_size;
    for (let i = 0; i < n; i++) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>x<sub>${i}</sub></td>`;

        data.results.forEach(result => {
            const td = document.createElement('td');
            td.textContent = result.normalized_solution[i].toFixed(4);
            tr.appendChild(td);
        });

        tbody.appendChild(tr);
    }
}

function renderMetrics(data) {
    elements.metricsGrid.innerHTML = '';

    // Best quantum method
    const quantumResults = data.results.filter(r => !r.name.includes('Classical'));
    const bestQuantum = quantumResults.reduce((best, r) =>
        r.final_cost < best.final_cost ? r : best, quantumResults[0]);

    // Classical reference
    const classical = data.results.find(r => r.name.includes('Classical'));

    // Total time
    const totalTime = data.results.reduce((sum, r) => sum + r.execution_time, 0);

    const metrics = [
        {
            label: 'Problem Size',
            value: `${data.problem_size}×${data.problem_size}`,
            class: 'info'
        },
        {
            label: 'Best Quantum Method',
            value: bestQuantum ? bestQuantum.name.split(' ')[0] : 'N/A',
            class: 'success'
        },
        {
            label: 'Best Quantum Error',
            value: bestQuantum ? bestQuantum.final_cost.toExponential(2) : 'N/A',
            class: bestQuantum && bestQuantum.final_cost < 1e-4 ? 'success' : 'warning'
        },
        {
            label: 'Classical Error',
            value: classical ? classical.final_cost.toExponential(2) : 'N/A',
            class: 'success'
        },
        {
            label: 'Total Compute Time',
            value: `${(totalTime * 1000).toFixed(0)} ms`,
            class: 'info'
        },
        {
            label: 'Methods Compared',
            value: data.results.length.toString(),
            class: 'info'
        }
    ];

    metrics.forEach(metric => {
        const div = document.createElement('div');
        div.className = 'metric-item';
        div.innerHTML = `
            <span class="metric-label">${metric.label}</span>
            <span class="metric-value ${metric.class}">${metric.value}</span>
        `;
        elements.metricsGrid.appendChild(div);
    });
}

// ============================================
// Event Listeners
// ============================================
function initEventListeners() {
    // Matrix size slider
    elements.matrixSize.addEventListener('input', (e) => {
        elements.matrixSizeValue.textContent = e.target.value;
        // Reset custom matrix when size changes
        if (customMatrixEnabled) {
            const n = parseInt(e.target.value);
            const forcePos = parseInt(elements.forcePosition.value);
            customMatrix = generateStiffnessMatrix(n);
            customVector = generateForceVector(n, forcePos);
        }
        updateVisualization();
    });

    // Force position
    elements.forcePosition.addEventListener('change', () => {
        if (!customMatrixEnabled) {
            updateVisualization();
        }
    });

    // Solve button
    elements.solveBtn.addEventListener('click', solveProblem);

    // Custom matrix toggle
    if (elements.useCustomMatrix) {
        elements.useCustomMatrix.addEventListener('change', (e) => {
            toggleCustomMatrix(e.target.checked);
        });
    }

    // Reset matrix button
    if (elements.resetMatrixBtn) {
        elements.resetMatrixBtn.addEventListener('click', resetToDefault);
    }

    // Chart scale buttons
    elements.chartBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            elements.chartBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            chartScale = e.target.dataset.scale;
            if (currentResults) {
                renderConvergenceChart(currentResults.results);
            }
        });
    });

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }

            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

// ============================================
// Initialization
// ============================================
function init() {
    updateVisualization();
    initEventListeners();

    // Check API health
    fetch(`${API_BASE_URL}/health`)
        .then(res => res.json())
        .then(data => {
            console.log('API Status:', data);
        })
        .catch(err => {
            console.warn('API not reachable. Make sure the backend is running.');
        });
}

// Run on DOM ready
document.addEventListener('DOMContentLoaded', init);
