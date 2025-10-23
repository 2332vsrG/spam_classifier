        const API_URL = 'http://localhost:5000/api';
        let classificationHistory = [];

        // Update character count
        document.getElementById('emailInput').addEventListener('input', function() {
            document.getElementById('charCount').textContent = this.value.length;
        });

        async function classifyEmail() {
            const emailText = document.getElementById('emailInput').value.trim();
            const errorMsg = document.getElementById('errorMessage');
            const loading = document.getElementById('loading');
            const resultSection = document.getElementById('resultSection');

            // Clear previous messages
            errorMsg.classList.remove('show');
            errorMsg.textContent = '';

            // Validation
            if (!emailText) {
                showError('Please enter email text to analyze');
                return;
            }

            if (emailText.length < 3) {
                showError('Email text is too short. Please enter a longer message.');
                return;
            }

            // Show loading
            loading.style.display = 'block';
            resultSection.style.display = 'none';

            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email: emailText })
                });

                if (!response.ok) {
                    throw new Error('Server error. Make sure backend is running on localhost:5000');
                }

                const data = await response.json();

                if (data.success) {
                    displayResult(data);
                    addToHistory(emailText, data.prediction, data.confidence);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            } catch (error) {
                showError('Cannot connect to backend. Ensure Flask app is running: python app.py');
                console.error('Error:', error);
            } finally {
                loading.style.display = 'none';
            }
        }

        function displayResult(data) {
            const resultSection = document.getElementById('resultSection');
            const resultBox = document.getElementById('resultBox');
            const resultLabel = document.getElementById('resultLabel');
            const resultBadge = document.getElementById('resultBadge');
            const resultSubtext = document.getElementById('resultSubtext');
            const confidencePercent = document.getElementById('confidencePercent');
            const confidenceFill = document.getElementById('confidenceFill');
            const probSpam = document.getElementById('probSpam');
            const probHam = document.getElementById('probHam');
            const modelAccuracy = document.getElementById('modelAccuracy');

            const isSpam = data.prediction === 'spam';
            const confidence = data.confidence * 100;

            // Update result
            resultLabel.textContent = isSpam ? 'SPAM' : 'LEGITIMATE';
            resultLabel.className = isSpam ? 'result-label label-spam' : 'result-label label-ham';
            resultSubtext.textContent = isSpam ? 'This email appears to be spam' : 'This email appears to be legitimate';

            // Update badge
            resultBadge.textContent = isSpam ? '⚠️ SPAM' : '✓ LEGITIMATE';
            resultBadge.className = isSpam ? 'badge badge-spam' : 'badge badge-ham';

            // Update box styling
            resultBox.className = isSpam ? 'result-box result-spam' : 'result-box result-ham';

            // Update confidence
            confidencePercent.textContent = confidence.toFixed(2) + '%';
            confidenceFill.style.width = confidence + '%';
            confidenceFill.className = isSpam ? 'progress-fill progress-spam' : 'progress-fill progress-ham';

            // Update probabilities
            probSpam.textContent = (data.probability_spam * 100).toFixed(2) + '%';
            probHam.textContent = (data.probability_ham * 100).toFixed(2) + '%';

            // Update model accuracy
            modelAccuracy.textContent = (data.model_accuracy * 100).toFixed(2) + '%';

            resultSection.style.display = 'block';
        }

        function addToHistory(email, prediction, confidence) {
            const preview = email.substring(0, 50) + (email.length > 50 ? '...' : '');
            classificationHistory.unshift({
                email: preview,
                prediction: prediction,
                confidence: confidence
            });

            if (classificationHistory.length > 10) {
                classificationHistory.pop();
            }

            updateHistoryDisplay();
        }

        function updateHistoryDisplay() {
            const historySection = document.getElementById('historySection');
            const historyList = document.getElementById('historyList');

            if (classificationHistory.length > 0) {
                historySection.style.display = 'block';
                historyList.innerHTML = classificationHistory.map((item, index) => `
                    <div class="history-item">
                        <span>${item.email}</span>
                        <span class="history-prediction ${item.prediction === 'spam' ? 'badge-spam' : 'badge-ham'}" style="background: ${item.prediction === 'spam' ? '#ff6b6b' : '#51cf66'}; color: white;">
                            ${item.prediction.toUpperCase()} (${(item.confidence * 100).toFixed(1)}%)
                        </span>
                    </div>
                `).join('');
            }
        }

        function clearInput() {
            document.getElementById('emailInput').value = '';
            document.getElementById('charCount').textContent = '0';
            document.getElementById('resultSection').style.display = 'none';
            document.getElementById('errorMessage').classList.remove('show');
        }

        function showError(message) {
            const errorMsg = document.getElementById('errorMessage');
            errorMsg.textContent = '❌ ' + message;
            errorMsg.classList.add('show');
        }

        // Check backend connection on load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch(`${API_URL}/health`);
                if (!response.ok) {
                    showError('⚠️ Backend not connected. Start with: python app.py');
                }
            } catch (error) {
                showError('⚠️ Cannot connect to backend. Ensure Flask is running: python app.py');
            }
        });