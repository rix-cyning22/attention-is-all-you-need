import React, { useState } from "react";

const StepDetail = ({ step, index, nextStep, prevStep, totalSteps, text }) => {
  return (
    <div className="bg-gray-800 p-4 my-4 rounded-lg shadow-md flex flex-col items-center w-1/2">
      <h1 className="text-xl font-semibold text-white mb-4">{text[index]}</h1>
      <ul className="text-base text-white w-3/4">
        {Object.entries(step).map(([word, probability], i) => (
          <li key={i} className="flex justify-around items-center py-1">
            <span>{word}</span>
            <span className="text-blue-500">{probability.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <div className="mt-6 flex justify-between w-full">
        {index > 0 && (
          <button
            className="bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 transition-colors duration-300"
            onClick={prevStep}
          >
            Previous
          </button>
        )}
        {index < totalSteps - 1 ? (
          <button
            className="bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 transition-colors duration-300 ml-auto"
            onClick={nextStep}
          >
            Next
          </button>
        ) : (
          <div className="ml-auto"></div>
        )}
      </div>
    </div>
  );
};

const ResultsTab = ({ result }) => {
  const [toggle, setToggle] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const textList = result.text.split(" ").slice(-result.words_gen);
  const nextStep = () => {
    setCurrentStep((prevStep) =>
      prevStep < result.prob_words.length - 1 ? prevStep + 1 : prevStep
    );
  };

  const prevStep = () => {
    setCurrentStep((prevStep) => (prevStep > 0 ? prevStep - 1 : prevStep));
  };

  return (
    <div className="p-4 bg-gray-600 flex flex-col items-center m-4 rounded-xl">
      <h1 className="text-lg">{result.text}</h1>
      <button
        className="text-blue-500 focus:outline-none"
        onClick={() => setToggle(!toggle)}
      >
        {toggle ? "Hide Details" : "Show Details"}
      </button>
      {toggle && result.prob_words.length > 0 && (
        <StepDetail
          step={result.prob_words[currentStep]}
          index={currentStep}
          nextStep={nextStep}
          prevStep={prevStep}
          totalSteps={result.prob_words.length}
          text={textList}
        />
      )}
    </div>
  );
};

export default ResultsTab;
