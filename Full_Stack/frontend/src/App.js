import React, { useState } from "react";

import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap/dist/css/bootstrap-grid.css";
import "bootswatch/dist/lux/bootstrap.min.css";
import FileUpload from "./components/FileUpload";

function App() {
  const [response, setResponse] = useState("");

  const handleSubmit = async (event) => {
    const formData = new FormData();
    console.log("handleUpload working");
    formData.append("question", event);

    try {
      const res = await fetch("http://localhost:8000/app/answer/", {
        method: "POST",
        body: formData,
      });

      const bd = await res.json();

      if (res.ok) {
        // File uploaded successfully
        console.log("Answer uploaded successfully");
        setResponse(bd.message);
        console.log(bd);
      } else {
        // Handle error if the file upload failed
        console.error("Answer upload failed");
      }
    } catch (error) {
      console.error("Error occurred during answer upload:", error);
    }
  };

  return (
    <>
      <div className="bg-light mb-4">
        <div className="container py-5">
          <h1 className="text-center mb-5">QA Bot</h1>
          <FileUpload onSubmit={handleSubmit} />
        </div>
      </div>

      {response && (
        <div className="card mx-auto" style={{ maxWidth: "1000px" }}>
          <div className="mt-3 mx-3 mb-3" style={{ fontWeight: "700" }}>
            {" "}
            Response :{" "}
          </div>
          <div
            className="card-body"
            style={{ maxHeight: "300px", overflow: "scroll" }}
          >
            {response}
          </div>
        </div>
      )}
    </>
  );
}

export default App;
