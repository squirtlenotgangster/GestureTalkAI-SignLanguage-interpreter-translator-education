/* =========================
   CONTACT PAGE SCRIPT (FIXED)
========================= */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".contact_form");
  const API_URL = "http://127.0.0.1:8000/contact"; // Change this if you deploy to Render!

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault(); // Stop page reload

    // 1. Get Values
    const firstName = form.querySelector('input[name="First Name"]').value.trim();
    const lastName = form.querySelector('input[name="Last Name"]').value.trim();
    const email = form.querySelector('input[name="Email Address"]').value.trim();
    const message = form.querySelector("textarea").value.trim();

    // 2. Validation
    if (!firstName || !lastName || !email || !message) {
      showAlert("Please fill in all fields.", "error");
      return;
    }

    if (!isValidEmail(email)) {
      showAlert("Please enter a valid email address.", "error");
      return;
    }

    // 3. Send to Backend
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: `${firstName} ${lastName}`, // Combine names for backend
          email: email,
          message: message,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        showAlert("Message sent successfully!", "success");
        form.reset();
      } else {
        showAlert("Error: " + (data.detail || "Something went wrong"), "error");
      }
    } catch (error) {
      console.error("Error:", error);
      showAlert("Failed to connect to the server.", "error");
    }
  });
});

/* =========================
   HELPERS
========================= */
function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function showAlert(message, type) {
  const oldAlert = document.querySelector(".form-alert");
  if (oldAlert) oldAlert.remove();

  const alert = document.createElement("div");
  alert.className = `form-alert ${type}`;
  alert.textContent = message;

  document.querySelector(".contact_form").prepend(alert);

  setTimeout(() => {
    alert.remove();
  }, 3500);
}