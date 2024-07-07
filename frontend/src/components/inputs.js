import React, { useState } from "react";

export const NumInput = ({ label, value, setValue }) => {
  const handleChange = (event) => {
    var value = event.target.value;
    if (value <= 0) value = 1;
    setValue(value);
  };
  return (
    <div className="flex space-y-2 justify-between items-center">
      <label className="text-lg">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(event) => handleChange(event)}
        className="p-2 rounded-md w-3/4 bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
      />
    </div>
  );
};

export const SeqInput = ({ label, seq, setSeq }) => {
  const [err, setErr] = useState("");
  const handleChange = (event) => {
    const value = event.target.value;
    if (value.trim().length === 0) {
      setErr("Sequence cannot be empty");
      setSeq("");
    } else {
      const words = value.split().filter((word) => word.length > 0);
      if (words.length > 5) {
        const newSeq = words.slice(words.length - 5).join(" ");
        setErr(`Sequence truncated to: "${newSeq}}"`);
        setSeq(newSeq);
      } else {
        setSeq(value);
        setErr("");
      }
    }
  };
  return (
    <div className="flex justify-between items-center">
      <label className="text-lg">{label}</label>
      <div className="w-3/4">
        <input
          type="text"
          value={seq}
          onChange={(event) => handleChange(event)}
          className="p-2 rounded-md w-full bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
        />
        {err.length > 0 && <small className="text-red">{err}</small>}
      </div>
      <div>
        <button className="rounded-md p-2 bg-blue-800">Submit</button>
      </div>
    </div>
  );
};
