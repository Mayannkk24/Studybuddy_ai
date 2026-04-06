# 🎓 StudyBuddy AI: The Ultimate Exam Saver
Let's be honest: Engineering syllabuses are huge, and old question papers (PYQs) are usually just messy PDFs or blurry photos. StudyBuddy AI is a smart tool I built to help students stop wasting time and start studying what actually matters.

It uses Gemini 2.5 Flash (the latest AI) to "read" your documents and tell you exactly what to focus on.

# 🚀 What can it do? (Features)
Syllabus to Notes: Upload a photo of your syllabus, and the AI writes detailed notes for every unit. It even marks topics as [HIGH PRIORITY] so you know what's important.

Exam Predictor: Feed it your Previous Year Questions (PYQs). It finds patterns and predicts which 10-mark questions are most likely to come this year.

Quick Cheat Sheet: Automatically creates a 1-page revision guide for last-minute study.

Self-Test Quiz: Generates a 5-question MCQ quiz after every analysis to see if you actually understood the topic.

Handwriting Reader: It can read text from clear photos, not just clean PDFs.

# 🛠️ Technical Architecture (The Stack)
I architected this using a modern AI Agent workflow:

Python: The main language for all the logic.

Google Gemini 2.5 Flash: The "Brain" that reads the images and understands the engineering concepts.

LangChain: This is the bridge that connects the AI to our app's specific goals.

RAG (In-Memory): A technique where the AI "retrieves" the specific text from your file to give accurate answers instead of just guessing.

Streamlit: Used for the modern "Dark Mode" website interface.

PyPDF2 & Pillow: To help the AI "see" and "read" PDFs and Images.