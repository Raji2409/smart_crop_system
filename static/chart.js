<canvas id="chart"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
var ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {
    type: 'line',
    data: {
        labels: {{ dates|safe }},
        datasets: [{
            label: 'Plant Health Timeline',
            data: {{ health|safe }}
        }]
    }
});
</script>
