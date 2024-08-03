const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
context.strokeStyle = 'black';
context.lineWidth = 10;
let drawing = false;

function getPosition(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (event.touches && event.touches[0]) {
        return {
            x: (event.touches[0].clientX - rect.left) * scaleX,
            y: (event.touches[0].clientY - rect.top) * scaleY
        };
    }

    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    drawing = true;
    const pos = getPosition(e);
    context.beginPath();
    context.moveTo(pos.x, pos.y);
}

function drawOnCanvas(e) {
    e.preventDefault();
    if (!drawing) return;
    const pos = getPosition(e);
    context.lineTo(pos.x, pos.y);
    context.stroke();
}

function stopDrawing() {
    drawing = false;
}

function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').textContent = 'Prediction will appear here';
    if (pieChart) {
        pieChart.destroy();
    }
}

function predict() {
    const image = canvas.toDataURL('image/png');
    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify({ image }),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = `Prediction: ${data.prediction}`;
        renderPieChart(data.probabilities);
    });
}

function renderPieChart(probabilities) {
    const ctx = document.getElementById('pieChart').getContext('2d');
    if (pieChart) {
        pieChart.destroy();
    }
    pieChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Banana', 'Basketball', 'Ladder'],
            datasets: [{
                data: probabilities,
                backgroundColor: ['#FFC300', '#FF5733', '#C70039']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw + '%';
                        }
                    }
                }
            }
        }
    });
}

let pieChart;

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', drawOnCanvas);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', drawOnCanvas);
canvas.addEventListener('touchend', stopDrawing);

// Prevent scrolling when touching the canvas
document.body.addEventListener("touchstart", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });
document.body.addEventListener("touchend", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });
document.body.addEventListener("touchmove", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });
