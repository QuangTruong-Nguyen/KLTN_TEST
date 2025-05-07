
"use client"
import { useRouter } from "next/navigation";
import { useSearchParams } from "next/navigation";
import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"

type Question = {
    question: string;
    option: string[];
    answer: string;
    explain: string;
    source: string;
  };

function QuizComponent({ quizData }: { quizData: any }) {
    const [answers, setAnswers] = useState<Record<number, string>>({});
    const [showResult, setShowResult] = useState(false);
    const searchParams = useSearchParams();
    const sessionId = searchParams.get("session_id");
    
    // Dữ liệu mẫu hoặc từ API
    const sampleData = {
      "questions": [
        {
          "question": "What is data mining?",
          "option": [
            "A) The process of collecting and analyzing data",
            "B) The process of searching and analyzing a large batch of raw data",
            "C) The process of transforming data into meaningful information",
            "D) The process of extracting insights from data"
          ],
          "answer": "B) The process of searching and analyzing a large batch of raw data",
          "explain": "Data mining is the process of searching and analyzing a large batch of raw data in order to identify patterns and extract useful information.",
          "source": "web"
        },
        {
          "question": "What is the goal of the data mining process?",
          "option": [
            "A) To collect and store data",
            "B) To analyze and extract insights from data",
            "C) To transform data into meaningful information",
            "D) To create profitable advertising and marketing campaigns"
          ],
          "answer": "B) To analyze and extract insights from data",
          "explain": "The goal of the data mining process is to study data mining from the perspective of different problem abstractions and data types.",
          "source": "database"
        },
        {
          "question": "What types of data are used in data mining?",
          "option": [
            "A) Quantitative and categorical data",
            "B) Text, spatial, and temporal data",
            "C) Graph-oriented and dependency-oriented data",
            "D) All of the above"
          ],
          "answer": "D) All of the above",
          "explain": "Data may be quantitative, categorical, text, spatial, temporal, or graph-oriented, with increasing volumes leading to data streams.",
          "source": "database"
        },
        {
          "question": "What is the data preprocessing phase in data mining?",
          "option": [
            "A) Feature extraction and data cleaning",
            "B) Data collection and feature extraction",
            "C) Data integration and clustering",
            "D) Analytical processing and algorithms"
          ],
          "answer": "A) Feature extraction and data cleaning",
          "explain": "The data preprocessing phase consists of feature extraction, data cleaning, and feature selection and transformation.",
          "source": "database"
        },
        {
          "question": "What is the purpose of feature extraction in data mining?",
          "option": [
            "A) To transform raw data into meaningful features",
            "B) To remove irrelevant features or transform the data space",
            "C) To handle erroneous or missing entries",
            "D) To determine similar customer groups"
          ],
          "answer": "A) To transform raw data into meaningful features",
          "explain": "Feature extraction transforms raw data into meaningful features, requiring domain understanding.",
          "source": "database"
        }
      ]
    };
  
    // Sử dụng dữ liệu từ API nếu có, nếu không dùng dữ liệu mẫu
    const data = quizData || sampleData;
  
    const handleAnswer = (index: number, value: string) => {
      setAnswers((prev) => ({ ...prev, [index]: value }));
    };
  
    const handleSubmit = () => {
      setShowResult(true);
    };
  
    const handleRetry = () => {
      setAnswers({});
      setShowResult(false);
    };
  
    const correctCount = data.questions.reduce((count: number, q: Question, index: number) => {
      return count + (answers[index] === q.answer ? 1 : 0);
    }, 0);
  
  
    const router = useRouter();
    return (
      
      <div className="p-4 max-w-3xl mx-auto space-y-4">
        {data.questions.map((q: Question, index: number) => (
          <Card key={index}>
            <CardContent className="p-4 space-y-2">
              <h2 className="font-semibold">{index + 1}. {q.question}</h2>
              <RadioGroup
                onValueChange={(val) => handleAnswer(index, val)}
                value={answers[index] || ""}
                disabled={showResult}
              >
                {q.option.map((opt, i) => (
                  <div key={i} className="flex items-center space-x-2">
                    <RadioGroupItem value={opt} id={`q${index}_opt${i}`} />
                    <label htmlFor={`q${index}_opt${i}`}>{opt}</label>
                  </div>
                ))}
              </RadioGroup>
  
              {showResult && (
                <div className="text-sm mt-2">
                  {answers[index] === q.answer ? (
                    <p className="text-green-500">✅ Chính xác!</p>
                  ) : (
                    <p className="text-red-500">❌ Sai. Đáp án đúng là: {q.answer}</p>
                  )}
                  <p className="text-gray-500">📘 Giải thích: {q.explain}</p>
                  <p className="text-gray-400 italic text-xs">Nguồn: {q.source}</p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
  
       
        <div className="mt-4 space-y-2">
          {!showResult ? (
            <Button onClick={handleSubmit} className="w-full">
              Nộp bài
            </Button>
          ) : (
            <div className="flex flex-col items-center gap-6">
              <div className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
                🎯 Bạn được {correctCount}/{data.questions.length} điểm
              </div>
              <Button
                onClick={handleRetry}
                variant="outline"
                className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:from-indigo-600 hover:to-purple-700"
              >
                Làm lại
              </Button>
            </div>
          )}
        </div>
      </div>    
    );
  }