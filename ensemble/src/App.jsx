import { useState } from 'react'
import './App.css'
import './components/glare-hover.css';
import GlareHover from './components/glare-hover.jsx';
import folderImg from "./assets/folder.png";
import folderOpenImg from "./assets/openedfolder.png"
import CircularText from './CircularText';


function setFolderOpen() {
  document.getElementById("openedFolder").src = {folderOpenImg};
}

function setFolderClose() {
  document.getElementById("Folder").src = {folderImg};
}

function App() { 
  const [hover, setHover] = useState(false);
  return (
    <div>
      <GlareHover>
        <input
        id="fileInput"
        type="file"
        style={{ display: "none" }}
        onChange={(e) => console.log(e.target.files)}
        />
        <label htmlFor='fileInput'>
          <img src ={folderImg} alt = "open your file" width={150}/>
        </label>
      </GlareHover>

      
    </div>
  );
}

export default App
