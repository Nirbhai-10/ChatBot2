document.addEventListener("DOMContentLoaded", () => {
  const fileUploadInput = document.getElementById("file-upload");
  const uploadButton = document.getElementById("upload-button");
  const textPasteInput = document.getElementById("text-paste");
  const pasteButton = document.getElementById("paste-button");
  const intentsList = document.getElementById("intents-list");
  const intentFilesList = document.getElementById("intent-files-list");
  const trainButton = document.getElementById("train-button");
  const progressBar = document.getElementById("progress-bar");

  let uploadedFiles = [];

  // Event listener for file upload
  fileUploadInput.addEventListener("change", () => {
    uploadedFiles = Array.from(fileUploadInput.files);
  });

  // Event listener for upload button
  uploadButton.addEventListener("click", () => {
    if (uploadedFiles.length > 0) {
      // Clear existing intent files list
      intentFilesList.innerHTML = "";
      progressBar.style.display = "block"; // Show the progress bar

      // Send uploaded files to server for conversion
      const formData = new FormData();
      uploadedFiles.forEach((file) => {
        formData.append("file", file);
      });

      fetch("/upload_intent", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.error) {
            console.error(data.error);
          } else {
            // Update intent files list
            data.intentFiles.forEach((intentFile) => {
              const listItem = document.createElement("li");
              listItem.textContent = intentFile;
              intentFilesList.appendChild(listItem);
            });
          }
          progressBar.style.display = "none"; // Hide the progress bar
        })
        .catch((error) => {
          console.error(error);
          progressBar.style.display = "none"; // Hide the progress bar in case of error
        });
    }
  });

  // Event listener for paste button
  pasteButton.addEventListener("click", () => {
    const text = textPasteInput.value.trim();
    if (text !== "") {
      // Clear existing intent files list
      intentFilesList.innerHTML = "";
      progressBar.style.display = "block"; // Show the progress bar

      // Send text to server for conversion
      fetch("/convert_text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.error) {
            console.error(data.error);
          } else {
            // Update intent files list
            data.intentFiles.forEach((intentFile) => {
              const listItem = document.createElement("li");
              listItem.textContent = intentFile;
              intentFilesList.appendChild(listItem);
            });
          }
          progressBar.style.display = "none"; // Hide the progress bar
        })
        .catch((error) => {
          console.error(error);
          progressBar.style.display = "none"; // Hide the progress bar in case of error
        });
    }
  });

  // Event listener for train button
  trainButton.addEventListener("click", () => {
    const selectedIntentFiles = Array.from(
      intentFilesList.getElementsByTagName("li")
    ).map((listItem) => listItem.textContent);

    if (selectedIntentFiles.length > 0) {
      // Show loading animation during training
      trainButton.disabled = true;
      trainButton.textContent = "Training...";
      progressBar.style.display = "block"; // Show the progress bar

      // Send selected intent files to server for training
      fetch("/train_chatbot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ intentFiles: selectedIntentFiles }),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            console.log("Chatbot trained successfully!");
          } else {
            console.error("Chatbot training failed.");
          }
          progressBar.style.display = "none"; // Hide the progress bar
          trainButton.disabled = false;
          trainButton.textContent = "Train"; // Reset the train button text
        })
        .catch((error) => {
          console.error(error);
          progressBar.style.display = "none"; // Hide the progress bar in case of error
          trainButton.disabled = false;
          trainButton.textContent = "Train"; // Reset the train button text
        });
    }
  });
});
