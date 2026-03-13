document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements selection
    const form = document.getElementById('prediction-form');
    const homeSelect = document.getElementById('home_team');
    const awaySelect = document.getElementById('away_team');
    const resultContainer = document.getElementById('result-container');
    const predictBtn = document.getElementById('predict-btn');

    /**
     * Updates the UI (logo and name) when a team is selected.
     * @param {HTMLElement} selectElement - The dropdown element.
     * @param {string} prefix - 'home' or 'away'.
     */
     function updateTeamDisplay(selectElement, prefix) {
        const teamName = selectElement.value;
        if (!teamName) return;

        document.getElementById(`${prefix}-name-display`).textContent = teamName;

        // Pointing to the local logos folder instead of the generic API
        // We assume the file name is Exactly "Team Name.png"
        const logoUrl = `/static/logos/${teamName}.png`;
        const imgElement = document.getElementById(`${prefix}-logo`);

        // Pulse animation
        imgElement.style.transform = 'scale(0.8)';

        // Attempt to load the local image
        imgElement.src = logoUrl;

        // Fallback: If the image is missing, use a placeholder
        imgElement.onerror = function() {
            this.src = `https://ui-avatars.com/api/?name=${teamName.replace(' ', '+')}&background=random&color=fff&size=100`;
        };

    setTimeout(() => {
        imgElement.style.transform = 'scale(1)';
    }, 150);
}

    // Event Listeners for dropdown changes
    homeSelect.addEventListener('change', () => updateTeamDisplay(homeSelect, 'home'));
    awaySelect.addEventListener('change', () => updateTeamDisplay(awaySelect, 'away'));

    // Form Submission using Async/Await Fetch
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevents full page reload

        const homeTeam = homeSelect.value;
        const awayTeam = awaySelect.value;

        // UI Feedback: Loading state
        const originalBtnText = predictBtn.innerText;
        predictBtn.innerHTML = '⚽ Analysing...';
        predictBtn.style.backgroundColor = '#ffc107';
        predictBtn.disabled = true;
        resultContainer.style.display = 'none';

        try {
            // Sending asynchronous POST request to Flask API
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam })
            });

            const data = await response.json();

            // Display result with slide-up animation
            resultContainer.innerHTML = data.error ? data.error : data.result;
            resultContainer.style.display = 'block';
            resultContainer.style.animation = 'none';
            setTimeout(() => resultContainer.style.animation = 'slideUp 0.5s ease-out', 10);

        } catch (error) {
            console.error("Error fetching prediction:", error);
            resultContainer.innerHTML = "⚠️ An error occurred while predicting.";
            resultContainer.style.display = 'block';
        } finally {
            // Revert button to original state
            predictBtn.innerHTML = originalBtnText;
            predictBtn.style.backgroundColor = '#00ff85';
            predictBtn.disabled = false;
        }
    });
});