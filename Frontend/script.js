const form = document.getElementById('predictionForm');
const button = document.getElementById('predictBtn');
const status = document.getElementById('status');
const resultArea = document.getElementById('resultArea');

async function loadOptions() {
    try {
        const response = await fetch('/options');
        if (!response.ok) throw new Error('Could not load property options. Refresh to try again.');
        const options = await response.json();
        for (const [field, id] of Object.entries({city: 'city', heating: 'heating', floor: 'Floor', usage_status: 'usage'})) {
            const select = document.getElementById(id);
            select.replaceChildren(new Option('Select an option', ''));
            for (const value of options[field]) select.add(new Option(value, value));
        }
        button.disabled = false;
    } catch (error) {
        status.textContent = error.message || 'Unable to load options. Refresh to try again.';
    }
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (button.disabled || !form.reportValidity()) return;
    button.disabled = true;
    button.textContent = 'Estimating…';
    status.textContent = '';
    resultArea.classList.add('hidden');
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            signal: controller.signal,
            body: JSON.stringify({
                city: document.getElementById('city').value,
                net_area: Number(document.getElementById('net_sqm').value),
                rooms: Number(document.getElementById('rooms').value),
                bathrooms: Number(document.getElementById('bathrooms').value),
                heating: document.getElementById('heating').value,
                floor: document.getElementById('Floor').value,
                total_floors: Number(document.getElementById('total_floors').value),
                building_age: Number(document.getElementById('age').value),
                usage_status: document.getElementById('usage').value
            })
        });
        if (!response.ok) {
            let message = `Prediction service returned ${response.status}. Please try again.`;
            if (response.status === 422) message = 'Check all fields and choose supported property values.';
            throw new Error(message);
        }
        const result = await response.json();
        if (!Number.isFinite(result.prediction) || result.prediction <= 0) throw new Error('The service returned an invalid estimate.');
        document.getElementById('priceText').textContent = new Intl.NumberFormat('tr-TR', {
            style: 'currency', currency: 'TRY', maximumFractionDigits: 0
        }).format(result.prediction);
        resultArea.classList.remove('hidden');
    } catch (error) {
        status.textContent = error.name === 'AbortError'
            ? 'The request timed out. Please try again.'
            : error.message || 'Unable to reach the prediction service. Please try again.';
    } finally {
        clearTimeout(timeout);
        button.disabled = false;
        button.textContent = 'Estimate Price';
    }
});

loadOptions();
