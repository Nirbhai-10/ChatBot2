window.addEventListener("DOMContentLoaded", () => {
    const inputFile = document.getElementById("input-file");
    const uploadButton = document.getElementById("upload-button");
    const conversionProgress = document.getElementById("conversion-progress");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");

    uploadButton.addEventListener("click", () => {
        const file = inputFile.files[0];
        if (!file) {
            alert("Please select a file");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        // Display conversion progress
        fileUpload.style.display = "none";
        conversionProgress.style.display = "block";

        // Send file to backend for conversion
        fetch("/upload_intent", {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                // Conversion completed
                progressText.textContent = data.message;
                progressBar.style.width = "100%";
            })
            .catch(error => {
                // Error occurred during conversion
                progressText.textContent = "Conversion failed";
                console.error(error);
            });
    });
});
