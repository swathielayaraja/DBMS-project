const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const bcrypt = require('bcrypt'); // for secure password hashing

const app = express();
const port = 2004;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Serve static files from the public directory
app.use(express.static('public'));

// MySQL database connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',             // MySQL username
    password: '#Sp@sql123',    // MySQL password
    database: 'NeuroSign'      // Database name
});

db.connect((err) => {
    if (err) {
        console.error('Database connection failed:', err.stack);
        return;
    }
    console.log('Connected to MySQL database');
});

// Route for user sign-up (registering a new user)
app.post('/registration', async (req, res) => {
    const { name, email, password } = req.body;

    // Hash the password before storing it
    const hashedPassword = await bcrypt.hash(password, 10);

    // Insert the new user into the database
    const query = 'INSERT INTO users (name, email, password) VALUES (?, ?, ?)';
    db.query(query, [name, email, hashedPassword], (err, results) => {
        if (err) {
            console.error('Error inserting user:', err);
            return res.status(500).json({ error: 'Database error' });
        }
        res.status(201).json({ message: 'User registered successfully' });
    });
});

// Route for user sign-in (logging in an existing user)
app.post('/signin', (req, res) => {
    const { email, password } = req.body;

    // Fetch the user by email
    const query = 'SELECT * FROM users WHERE email = ?';
    db.query(query, [email], async (err, results) => {
        if (err) {
            console.error('Error fetching user:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        // Check if user exists
        if (results.length === 0) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        const user = results[0];

        // Verify the password
        const passwordMatch = await bcrypt.compare(password, user.password);
        if (!passwordMatch) {
            return res.status(401).json({ error: 'Invalid email or password' });
        }

        // Successful login
        res.status(200).json({ message: 'Logged in successfully!' });
    });
});

// Define specific routes for each HTML page
app.get('/homepage', (req, res) => {
    res.sendFile(__dirname + '/public/homepage.html');
});

app.get('/chatgpt', (req, res) => {
    res.sendFile(__dirname + '/public/chatgpt.html');
});

app.get('/registration', (req, res) => {
    res.sendFile(__dirname + '/public/registration.html');
});

app.get('/signin', (req, res) => {
    res.sendFile(__dirname + '/public/signin.html'); // Ensure this matches the HTML file name
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
