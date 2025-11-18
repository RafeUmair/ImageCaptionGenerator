import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const dropRef = useRef(null);

  const handleFileSelect = (file) => {
    setFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    dropRef.current.classList.add("drag-over");
  };

  const handleDragLeave = () => {
    dropRef.current.classList.remove("drag-over");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dropRef.current.classList.remove("drag-over");

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) handleFileSelect(droppedFile);
  };

  const uploadImage = async () => {
    if (!file) return alert("Please upload an image first.");

    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8080/api/caption", {
        method: "POST",
        body: formData,
      });

      const data = await res.text();
      setCaption(data);
    } catch (err) {
      console.error(err);
      alert("Failed to generate caption");
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      <div className="card">

        <h2 className="title">Image Caption Generator</h2>

        <div
          className="drop-zone"
          ref={dropRef}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/png, image/jpeg, image/jpg"
            onChange={handleFileChange}
            className="file-input"
          />

          <p className="drop-text">
            {file ? <strong>{file.name}</strong> : <><strong>
              Click or Drag & Drop</strong> to upload image</>}
          </p>
          <p className="formats">PNG • JPG • JPEG</p>
        </div>

        {preview && (
          <div className="preview-container">
            <img className="preview-img" src={preview} alt="Preview" />
          </div>
        )}

        <button onClick={uploadImage} disabled={loading} className="btn">
          {loading ? "Generating..." : "Generate Caption"}
        </button>

        {caption && <div className="caption-box">{caption}</div>}
      </div>
    </div>
  );
}

export default App;
