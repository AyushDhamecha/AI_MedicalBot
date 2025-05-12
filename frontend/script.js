document.getElementById("getStartedButton").addEventListener("click", function() {
    const userMessage = prompt("Enter your message for Medibot:");

    if (userMessage) {
        fetch("http://localhost:5000/chat", {  // Connect to Flask backend
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: userMessage })
        })
        .then(response => response.json())
        .then(data => {
            alert("Medibot says: " + data.response);
        })
        .catch(error => console.error("Error:", error));
    }
});
