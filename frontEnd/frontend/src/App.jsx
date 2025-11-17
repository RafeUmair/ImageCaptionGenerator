import { useState } from "react";
import "./App.css";

function App() 
{
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null); 
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => 
    {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    if (selectedFile) 
      {
      setPreview(URL.createObjectURL(selectedFile));
    } 
    else 
      {
      setPreview(null);
    }
  };

  const uploadImage = async () => 
    {
    if (!file) return alert("Please select an image.");

    const formData = new FormData();
    formData.append("image", file); 

    setLoading(true);

    try 
    {
      const res = await fetch("http://localhost:8080/api/caption", {
        method: "POST",
        body: formData,
      });

      const data = await res.text();
      setCaption(data);
    }
     catch (err) 
     {
      console.error(err);
      alert("Failed to get caption");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 30 }}>
      <h2>🖼 Image Caption Generator</h2>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      {preview && (
        <div style={{ marginTop: 20 }}>
          <img
            src={preview}
            alt="Preview"
            style={{ maxWidth: "300px", maxHeight: "300px", border: "1px solid #ccc" }}
          />
        </div>
      )}

      <button onClick={uploadImage} disabled={loading} style={{ marginTop: 10 }}>
        {loading ? "Generating..." : "Generate Caption"}
      </button>

      {caption && (
        <pre style={{ background: "#000000ff", padding: 10, marginTop: 20 }}>
          {caption}
        </pre>
      )}
    </div>
  );
}

export default App;
