// static/script.js

document.getElementById('diabetes-form').addEventListener('submit', function (event) {
    event.preventDefault();
    const form = event.target;

    const formData = new FormData(form);

    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const outcome = result['Predicted Outcome'];
        let resultText = '';

        if (outcome === 0) {
            resultText = 'You are Safe But Stay Alert!';
        } else {
            resultText = 'Take Precautions and Take Steps Toward a Healthy Lifestyle';
        }

        const resultElement = document.getElementById('resultText');
        resultElement.textContent = resultText;
        resultElement.classList.add(outcome === 0 ? 'safe-outcome' : 'precautions-outcome');
    })
    .catch(error => console.error('Error:', error));
});
