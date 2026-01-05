async function analyzeTraffic() {
    // 1. Get the button and show loading
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Analyzing...';

    try {
        // 2. Call the backend (Using the mode selected in the UI)
        const response = await fetch(`/detect/${selectedMode}`, { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) throw new Error("Backend Error");

        const data = await response.json();

        // 3. Switch View
        document.getElementById('upload-section').style.display = 'none';
        document.getElementById('live-controls').style.display = 'none';
        document.getElementById('results-section').style.display = 'block';

        // 4. Update the Dashboard (The functions below are already in your file)
        displayResults(data);

    } catch (error) {
        console.error(error);
        alert("System Error: Could not connect to the IDS models.");
        btn.disabled = false;
        btn.innerHTML = 'Start Analysis';
    }
}