/* =========================
   LOGIN & REGISTER SCRIPT (FIXED)
========================= */

const container = document.querySelector('.container');
const registerBtn = document.querySelector('.register-btn');
const loginBtn = document.querySelector('.login-btn');

// Toggle Animation
registerBtn.addEventListener('click', () => {
  container.classList.add('active');
});

loginBtn.addEventListener("click", () => {
  container.classList.remove('active');
});

// --- BACKEND CONNECTION ---
const API_BASE = "http://127.0.0.1:8000"; // Change if deployed

// 1. Handle Registration
const registerForm = document.getElementById('registerForm'); // MAKE SURE YOU ADD THIS ID TO HTML!
if (registerForm) {
  registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Grab inputs (Update 'name' attributes if yours are different)
    const username = registerForm.querySelector('input[type="text"]').value;
    const email = registerForm.querySelector('input[type="email"]').value;
    const password = registerForm.querySelector('input[type="password"]').value;

    try {
      const response = await fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password })
      });

      const data = await response.json();

      if (response.ok) {
        alert("Registration Successful! Please Login.");
        container.classList.remove('active'); // Switch to login view
        registerForm.reset();
      } else {
        alert("Registration Failed: " + (data.detail || "Unknown error"));
      }
    } catch (err) {
      alert("Cannot connect to server.");
    }
  });
}

// 2. Handle Login
const loginForm = document.getElementById('loginForm'); // MAKE SURE YOU ADD THIS ID TO HTML!
if (loginForm) {
  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = loginForm.querySelector('input[type="text"]').value;
    const password = loginForm.querySelector('input[type="password"]').value;

    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });

      const data = await response.json();

      if (response.ok) {
        // SUCCESS! Redirect the user
        // Adjust path based on your folder structure
        window.location.href = "../Landingpage/home.html"; 
      } else {
        alert("Login Failed: " + (data.detail || "Invalid credentials"));
      }
    } catch (err) {
      alert("Cannot connect to server.");
    }
  });
}