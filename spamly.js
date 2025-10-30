const testExamples = [
    "CONGRATULATIONS! You've been selected as our lucky winner. Click here to claim your $5000 prize before it expires!",
    "URGENT: Your account has been compromised. Verify your identity immediately by clicking this link and entering your password.",
    "Hi team, please review the attached document before tomorrow's meeting. Let me know if you have any questions.",
    "Hey! Are we still on for lunch tomorrow? Let me know what time works for you.",
    "Your bank statement is attached to this email. Please review it and let me know if you have any questions. Click the link to view it."
];

function loadExample(index) {
    document.getElementById('textInput').value = testExamples[index];
}

function clearAll() {
    document.getElementById('textInput').value = '';
    document.getElementById('resultsArea').classList.remove('active');
}

async function analyze() {
    const text = document.getElementById('textInput').value.trim();
    
    if (!text) {
        Swal.fire({
            icon: 'warning',
            title: 'No text entered',
            text: 'Please enter some text to analyze'
        });
        return;
    }

    // Show results section
    document.getElementById('resultsArea').classList.add('active');

    // calling the backend to get the predictions
    const predictions = await analyzeBackend(text);
    showResults(predictions);
}

async function analyzeBackend(text) {
    try {
        const response = await fetch('https://spamly-backend.onrender.com/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status} ${response.statusText} - ${errText}`);
        }
        let data;
        try {
            data = await response.json();
        } catch (e) {
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'Invalid JSON in response'
            });
            throw new Error('Invalid JSON in response');
        }
        return data;
    } catch (error) {
        console.error('Error analyzing message:', error);
        Swal.fire({
            icon: 'error',
            title: 'Oops...',
            text: 'Failed to analyze message. Please try again.'
        });
        return null;
    }
}


function showResults(data) {
    if (!data) return;

    // helper to normalize label to 'spam' | 'ham'
    const normalizeLabel = (raw) => {
        if (raw === 1 || raw === '1') return 'spam';
        if (raw === 0 || raw === '0') return 'ham';
        const s = String(raw).toLowerCase();
        return s === 'spam' ? 'spam' : 'ham';
    };

    // helper to normalize confidence to 0..100 number
    const toPercent = (c) => {
        let n = Number(c);
        if (!isFinite(n)) return 0;
        // if 0..1 scale, convert to percent; if already percent, clamp
        if (n <= 1) n = n * 100;
        return Math.max(0, Math.min(100, Math.round(n)));
    };

    // Naive Bayes
    const nbLabel = normalizeLabel(data.naive_bayes?.label);
    const nbConf = toPercent(data.naive_bayes?.confidence);

    document.getElementById('nb-result').textContent = nbLabel.toUpperCase();
    document.getElementById('nb-result').className = `result-label is-${nbLabel}`;
    document.getElementById('nb-confidence').textContent = `Confidence: ${nbConf}%`;
    document.getElementById('nb-bar').className = `confidence-bar ${nbLabel}-bar`;
    document.getElementById('nb-bar').style.width = `${nbConf}%`;

    // Logistic Regression
    const lrLabel = normalizeLabel(data.logistic_regression?.label);
    const lrConf = toPercent(data.logistic_regression?.confidence);

    document.getElementById('lr-result').textContent = lrLabel.toUpperCase();
    document.getElementById('lr-result').className = `result-label is-${lrLabel}`;
    document.getElementById('lr-confidence').textContent = `Confidence: ${lrConf}%`;
    document.getElementById('lr-bar').className = `confidence-bar ${lrLabel}-bar`;
    document.getElementById('lr-bar').style.width = `${lrConf}%`;

    // Summary
    let summaryText = '';
    if (nbLabel === lrLabel) {
        const agreed = nbLabel === 'spam' ? 'SPAM' : 'HAM';
        summaryText = `Both models agree that this message is <strong>${agreed}</strong>. `;
        summaryText += nbLabel === 'spam'
            ? 'This message shows characteristics typical of spam or phishing attempts.'
            : 'This appears to be a legitimate message.';
    } else {
        summaryText = `The models disagree on this one. Naive Bayes says <strong>${nbLabel.toUpperCase()}</strong> while Logistic Regression says <strong>${lrLabel.toUpperCase()}</strong>. The message might be borderline or contain mixed signals.`;
    }
    document.getElementById('summary').innerHTML = summaryText;
}