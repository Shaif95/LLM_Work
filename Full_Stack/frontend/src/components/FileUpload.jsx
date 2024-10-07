import React, { useState } from "react";
import Button from "react-bootstrap/Button";

function FileUpload({ onSubmit }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [textInput, setTextInput] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    console.log("rahil");
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    if (selectedFile) {
      const formData = new FormData();
      console.log("handleUpload working");
      formData.append("myfile", selectedFile);

      try {
        const response = await fetch("http://localhost:8000/app/upload/", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          // File uploaded successfully
          console.log("File uploaded successfully");
        } else {
          // Handle error if the file upload failed
          console.error("File upload failed");
        }
      } catch (error) {
        console.error("Error occurred during file upload:", error);
      }
    }
  };

  const handleTextChange = (event) => {
    setTextInput(event.target.value);
  };

  const handleSubmit = () => {
    onSubmit(textInput);
  };

  return (
    <div className="card mx-auto" style={{ maxWidth: "600px" }}>
      <div className="card-body">
        <div className="mb-3">
          <label htmlFor="fileInput" className="form-label">
            Choose a file:
          </label>
          <div className="d-flex flex-row">
            <input
              type="file"
              id="fileInput"
              onChange={handleFileChange}
              className="form-control me-3"
            />
            <Button variant="primary" onClick={handleUpload} className="w-10">
              Upload
            </Button>
          </div>
        </div>

        <div className="mb-3">
          <label htmlFor="textInput" className="form-label">
            Enter some text:
          </label>
          <input
            type="text"
            id="textInput"
            value={textInput}
            placeholder="search"
            onChange={handleTextChange}
            className="form-control"
          />
        </div>
        <Button variant="primary" onClick={handleSubmit} className="w-10">
          Submit
        </Button>
      </div>
    </div>
  );
}

export default FileUpload;
