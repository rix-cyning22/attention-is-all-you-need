import axios from "axios";
import { useState } from "react";
import { SeqInput, NumInput } from "./inputs";

const InputBar = ({ setGenSeq }) => {
  const [seq, setSeq] = useState("Sample text");
  const [gen, setGen] = useState(4);
  const [prob, setProb] = useState(5);
  const [toggle, setToggle] = useState(false);
  const handleSubmit = async (event) => {
    event.preventDefault();
    const res = await axios.post("http://localhost:5000/api", {
      seq: seq,
      gen_seq_length: gen,
      get_num_words: prob,
    });
    setGenSeq(res.data);
  };
  return (
    <form
      className="flex flex-col p-6 bg-gray-800 text-white space-y-2 shadow-lg w-full fixed bottom-0"
      onSubmit={(event) => handleSubmit(event)}
    >
      <button
        className="rounded-md p-2 bg-black text-white ml-auto"
        onClick={() => setToggle(!toggle)}
      >
        {toggle ? "Hide Settings" : "Show Settings"}
      </button>
      <SeqInput label="Enter a 5-word text" seq={seq} setSeq={setSeq} />
      {toggle && (
        <div className="w-full">
          <NumInput
            label="Generated Sequence Length"
            value={gen}
            setValue={setGen}
          />
          <NumInput label="Probable Words" value={prob} setValue={setProb} />
        </div>
      )}
    </form>
  );
};

export default InputBar;
