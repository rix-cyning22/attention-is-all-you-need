import { useState } from "react";
import "./App.css";
import InputBar from "./components/inputbar";
import ResultsTab from "./components/results";

function App() {
  const [genSeq, setGenSeq] = useState({});
  return (
    <div className="App">
      <div className="bg-black min-h-screen text-white">
        {Object.keys(genSeq).length > 0 && <ResultsTab result={genSeq} />}
        <InputBar setGenSeq={setGenSeq} />
      </div>
    </div>
  );
}

export default App;
