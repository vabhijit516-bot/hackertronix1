// Unified Auto-Detect Web Dashboard Script
document.addEventListener('DOMContentLoaded', () => {
    // Task 1 Controls Sync
    const ballToggle = document.getElementById('ball-toggle');
    const engineSelect = document.getElementById('engine-select');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const engineTag = document.getElementById('engine-tag');

    if (ballToggle) {
        ballToggle.addEventListener('change', (e) => {
            updateConfig({ enable_ball_detection: e.target.checked });
        });
    }

    confSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value).toFixed(2);
        confVal.textContent = val;
        updateConfig({ conf_thresh: val });
    });

    engineSelect.addEventListener('change', (e) => {
        const mode = e.target.value;
        engineTag.textContent = mode === 'hough' ? 'HSV Hough Circles' : 'YOLOv8 ONNX';
        updateConfig({ engine: mode });
    });

    // Task 2 Controls Sync
    const focalSlider = document.getElementById('focal-slider');
    const focalVal = document.getElementById('focal-val');
    const filterSelect = document.getElementById('filter-select');

    focalSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        focalVal.textContent = `${val} px`;
        updateConfig({ focal_px: val });
    });

    filterSelect.addEventListener('change', (e) => {
        updateConfig({ filter_type: e.target.value });
    });

    function updateConfig(payload) {
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).catch(err => console.error('Error updating config:', err));
    }

    // Quick Calibration Button
    const btnCalibrate = document.getElementById('btn-calibrate');
    const calibWPx = document.getElementById('calib-w-px');
    const calibZM = document.getElementById('calib-z-m');

    btnCalibrate.addEventListener('click', () => {
        const w_px = parseFloat(calibWPx.value);
        const z_m = parseFloat(calibZM.value);

        fetch('/api/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ w_px, z_m })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                const newFocal = Math.round(data.focal_px);
                focalSlider.value = newFocal;
                focalVal.textContent = `${newFocal} px`;
                alert(`✅ Calibration Success! New Focal Length: ${newFocal} px`);
            } else {
                alert(`❌ Calibration Error: ${data.message}`);
            }
        })
        .catch(err => alert(`Calibration failed: ${err}`));
    });

    // Live Telemetry Poller
    function pollTelemetry() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                document.getElementById('fps-val').textContent = (data.fps || 0.0).toFixed(1);
                document.getElementById('balls-count').textContent = data.balls_count || 0;
                document.getElementById('face-z').textContent = `${data.last_z || 0.00} m`;
                document.getElementById('face-angle').textContent = `${data.last_angle || 0.0}°`;
                if (data.engine && engineTag) {
                    engineTag.textContent = data.engine;
                }
            })
            .catch(err => console.error('Telemetry error:', err));
    }

    setInterval(pollTelemetry, 1000);
});
