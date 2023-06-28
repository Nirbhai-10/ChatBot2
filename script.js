function sendMessage() {
  var userInput = document.getElementById("user-input").value;
  document.getElementById("user-input").value = "";

  var chatMessages = document.getElementById("chat-messages");
  var userMessage = document.createElement("div");
  userMessage.classList.add("message", "user");
  userMessage.innerHTML = "<strong>You:</strong> " + userInput;
  chatMessages.appendChild(userMessage);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Send message to the chatbot API
  fetch("/ChatApp", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message: userInput }),
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('Request failed with status ' + response.status);
      }
      return response.json();
    })
    .then(data => {
      var botMessage = document.createElement("div");
      botMessage.classList.add("message", "bot");
      botMessage.innerHTML = "<strong>ChatBot:</strong> " + data.response;
      chatMessages.appendChild(botMessage);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    })
    .catch(error => {
      console.error("Error:", error);
    });
}

// function sendMessage() {
//   var userInput = document.getElementById("user-input").value;
//   document.getElementById("user-input").value = "";

//   var chatMessages = document.getElementById("chat-messages");
//   var userMessage = document.createElement("div");
//   userMessage.classList.add("message", "user");
//   userMessage.innerHTML = "<strong>You:</strong> " + userInput;
//   chatMessages.appendChild(userMessage);
//   chatMessages.scrollTop = chatMessages.scrollHeight;

//   // Send message to the chatbot API
//   fetch("https://8fa5-122-162-151-37.ngrok-free.app/ChatApp", {
//     method: "POST",
//     mode: "no-cors",
//     headers: {
//       "Content-Type": "application/json; charset=utf-8"
//     },
    
//     body: JSON.stringify({ message: userInput }),
//   })
//     .then(response => response.json())
//     .then(data => {
//       var botMessage = document.createElement("div");
//       botMessage.classList.add("message", "bot");
//       botMessage.innerHTML = "<strong>ChatBot:</strong> " + data.response;
//       chatMessages.appendChild(botMessage);
//       chatMessages.scrollTop = chatMessages.scrollHeight;
//     })
//     .catch(error => {
//       console.error("Error:", error);
//     });
// }
