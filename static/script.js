document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const button = document.querySelector('button');
    const resultDiv = document.querySelector('.result');

    if (resultDiv) {
        resultDiv.style.opacity = '0';
        resultDiv.style.transition = 'opacity 0.8s ease-in-out';
        setTimeout(() => {
            resultDiv.style.opacity = '1';
        }, 100);
    }

    form.addEventListener('submit', function() {
        button.innerHTML = '⚽ Analysing Stats...';
        button.style.backgroundColor = '#ffc107';
        button.style.cursor = 'not-allowed';
    });
});