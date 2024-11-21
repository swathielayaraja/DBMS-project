const express = require('express');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = 4000;

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Route to run the Python script
app.get('/run-python', (req, res) => {
    exec('python app.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing Python script: ${error.message}`);
            return res.status(500).send('Server error');
        }
        if (stderr) {
            console.error(`Python stderr: ${stderr}`);
            return res.status(500).send('Python script error');
        }
        
        // Send Python script output back to client
        res.send(`<pre>${stdout}</pre>`);
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
