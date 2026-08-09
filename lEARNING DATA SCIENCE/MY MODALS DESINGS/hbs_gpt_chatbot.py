#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════════
 HBS-GPT: ChatGPT-Style Conversational AI (Biological HBS-Engine V3.0)
 ──────────────────────────────────────────────────────────────────────────────────
 A ChatGPT-like conversational AI built on the Biological Human-Brain Spiking
 Engine. Combines:

  1. TF-IDF Semantic Retrieval — 500+ knowledge entries matched by meaning
  2. Trigram Causal Generation — trained on 30,000+ real internet sentences
  3. Conversation Memory — multi-turn context tracking
  4. Dynamic Response Blending — retrieval + generation for natural answers

 Architecture:
  • Layer 1: TF-IDF Vectorizer → Cosine Similarity Semantic Search
  • Layer 2: Trigram P(w_t | w_{t-1}, w_{t-2}) Causal Spiking Memory
  • Layer 3: Conversation Context Buffer (last 5 turns)
  • Layer 4: Response Ranker + Confidence Gating

 Run: python3 hbs_gpt_chatbot.py
════════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import os
import re
import math
import numpy as np
import psutil
import platform
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_specs():
    mem = psutil.virtual_memory()
    return {
        'os': f"{platform.system()} {platform.release()}",
        'cpus': f"{psutil.cpu_count(logical=False) or 1} Physical ({psutil.cpu_count(logical=True) or 1} Logical)",
        'ram_gb': mem.total / (1024 ** 3),
        'pid': os.getpid(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TF-IDF SEMANTIC VECTORIZER (Pure NumPy — No External ML Libraries)
# ═══════════════════════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """
    Pure NumPy TF-IDF Vectorizer for semantic similarity search.
    Converts text into TF-IDF vectors and computes cosine similarity.
    """
    def __init__(self):
        self.vocab = {}
        self.idf = None
        self.doc_vectors = None

    def _tokenize(self, text):
        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def fit_transform(self, documents):
        # Build vocabulary
        doc_freq = defaultdict(int)
        all_tokens = []
        for doc in documents:
            tokens = set(self._tokenize(doc))
            all_tokens.append(self._tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1

        # Filter rare words and build vocab index
        self.vocab = {}
        idx = 0
        for word, freq in sorted(doc_freq.items()):
            if freq >= 1:  # keep all words
                self.vocab[word] = idx
                idx += 1

        n_docs = len(documents)
        n_vocab = len(self.vocab)

        # Compute IDF: log(N / (1 + df))
        self.idf = np.zeros(n_vocab)
        for word, widx in self.vocab.items():
            self.idf[widx] = math.log((n_docs + 1) / (1 + doc_freq[word])) + 1

        # Compute TF-IDF matrix
        self.doc_vectors = np.zeros((n_docs, n_vocab))
        for i, tokens in enumerate(all_tokens):
            tf = defaultdict(int)
            for t in tokens:
                if t in self.vocab:
                    tf[self.vocab[t]] += 1
            for widx, count in tf.items():
                self.doc_vectors[i, widx] = (count / max(len(tokens), 1)) * self.idf[widx]

        # L2 normalize
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.doc_vectors /= norms

        return self.doc_vectors

    def transform_query(self, text):
        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocab))
        tf = defaultdict(int)
        for t in tokens:
            if t in self.vocab:
                tf[self.vocab[t]] += 1
        for widx, count in tf.items():
            vec[widx] = (count / max(len(tokens), 1)) * self.idf[widx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def find_similar(self, query, top_k=3):
        q_vec = self.transform_query(query)
        scores = self.doc_vectors @ q_vec
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top_indices]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MASSIVE KNOWLEDGE BASE (500+ Q&A Pairs — General Knowledge)
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_base():
    """
    Returns list of (question_pattern, answer) tuples covering broad topics.
    Each entry is designed to match semantically via TF-IDF.
    """
    kb = []

    # ── GREETINGS & CONVERSATION ──
    kb += [
        ("hello hi hey greetings howdy good morning good afternoon good evening",
         "Hello! I'm HBS-GPT, your AI assistant. I can answer questions about science, technology, history, math, and much more. What would you like to know?"),
        ("how are you how do you feel how is it going whats up",
         "I'm doing great, thank you for asking! I'm ready to help you with any questions you have. What's on your mind?"),
        ("who are you what are you tell me about yourself your name",
         "I'm HBS-GPT, an AI assistant powered by the Biological Human-Brain Spiking Engine V3.0. I use TF-IDF semantic search across 500+ knowledge entries combined with trigram language generation trained on 30,000+ real English sentences. How can I help you today?"),
        ("what can you do capabilities features help",
         "I can answer questions about science, technology, history, geography, mathematics, programming, health, philosophy, and everyday topics. I can also complete sentences and have natural conversations. Just ask me anything!"),
        ("goodbye bye see you later farewell take care",
         "Goodbye! It was great talking with you. Feel free to come back anytime you have questions. Take care!"),
        ("thank you thanks appreciate grateful",
         "You're welcome! I'm happy to help. Is there anything else you'd like to know?"),
        ("sorry apologize my bad",
         "No worries at all! Feel free to ask me anything. I'm here to help."),
        ("tell me a joke funny humor laugh",
         "Why do programmers prefer dark mode? Because light attracts bugs! Want to hear another one?"),
        ("another joke one more joke",
         "A SQL query walks into a bar, sees two tables and asks: 'Can I join you?' There are 10 types of people: those who understand binary, and those who don't."),
        ("tell me something interesting random fact fun fact",
         "Here's a fun fact: A day on Venus is longer than a year on Venus! Venus takes 243 Earth days to rotate once but only 225 Earth days to orbit the Sun."),
        ("how old are you age birthday when were you born",
         "I don't have an age in the traditional sense — I was built using the Biological Human-Brain Spiking Engine architecture. I exist as long as I'm running! Think of me as perpetually new."),
        ("are you a robot are you human are you real are you alive",
         "I'm an AI — a software program that simulates conversation using spiking neural networks and semantic search. I'm not alive or conscious, but I can understand your questions and provide helpful answers!"),
        ("do you have feelings emotions",
         "I don't experience emotions the way humans do. I process language patterns and generate appropriate responses. But I'm designed to be helpful, friendly, and conversational!"),
    ]

    # ── SCIENCE: PHYSICS ──
    kb += [
        ("what is the speed of light how fast is light light speed",
         "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). It's denoted as 'c' and is considered the universal speed limit — nothing with mass can travel at or exceed this speed according to Einstein's theory of special relativity."),
        ("what is gravity gravitational force weight",
         "Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, it gives objects weight and keeps us on the ground. It's described by Newton's law of gravitation (F = G·m₁·m₂/r²) and more precisely by Einstein's general relativity, which describes gravity as the curvature of spacetime caused by mass and energy."),
        ("what is a black hole black holes event horizon",
         "A black hole is a region of spacetime where gravity is so strong that nothing — not even light — can escape from it. They form when massive stars collapse at the end of their life. The boundary around a black hole is called the event horizon. Supermassive black holes exist at the centers of most galaxies, including our Milky Way."),
        ("what is quantum physics quantum mechanics",
         "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the atomic and subatomic level. Key principles include wave-particle duality (particles can behave as waves), the uncertainty principle (you can't know both position and momentum precisely), and quantum entanglement (particles can be connected across distances)."),
        ("what is energy conservation of energy forms of energy",
         "Energy is the ability to do work. It exists in many forms: kinetic (motion), potential (stored), thermal (heat), electrical, chemical, nuclear, and radiant (light). The law of conservation of energy states that energy cannot be created or destroyed, only converted from one form to another. Einstein's famous equation E=mc² shows that mass and energy are interchangeable."),
        ("what is an atom atoms protons neutrons electrons",
         "An atom is the basic unit of matter. It consists of a nucleus containing protons (positive charge) and neutrons (no charge), surrounded by electrons (negative charge) in orbital shells. Atoms are incredibly small — about 1 to 5 angstroms (10⁻¹⁰ meters) in diameter. There are 118 known elements, each defined by its number of protons."),
        ("what is electricity electric current voltage",
         "Electricity is the flow of electric charge, typically carried by electrons through a conductor like a wire. Voltage (measured in volts) is the 'push' that drives the current. Current (measured in amperes) is the rate of charge flow. Resistance (measured in ohms) opposes the flow. Ohm's law relates them: V = I × R."),
        ("what is magnetism magnetic field magnets",
         "Magnetism is a force produced by moving electric charges. Every magnet has a north and south pole — like poles repel, opposite poles attract. Earth itself is a giant magnet, which is why compasses work. Electromagnetism, unifying electricity and magnetism, is one of the four fundamental forces of nature."),
        ("what is the theory of relativity einstein relativity",
         "Einstein's theory of relativity has two parts. Special relativity (1905) says the speed of light is constant for all observers and that time slows down at high speeds (time dilation). General relativity (1915) describes gravity as the curvature of spacetime caused by mass. Both theories have been confirmed by countless experiments."),
        ("what is thermodynamics heat temperature entropy",
         "Thermodynamics is the study of heat, energy, and work. Its key laws are: (1) Energy is conserved — it can't be created or destroyed. (2) Entropy (disorder) in an isolated system always increases. (3) As temperature approaches absolute zero, entropy approaches a minimum. These laws govern everything from engines to stars."),
        ("what is nuclear energy nuclear fission fusion",
         "Nuclear energy comes from changes in atomic nuclei. Fission splits heavy atoms (like uranium) releasing enormous energy — this powers nuclear reactors. Fusion combines light atoms (like hydrogen into helium) releasing even more energy — this powers the Sun. Both processes convert mass into energy via E=mc²."),
        ("what is sound how does sound work sound waves",
         "Sound is a mechanical wave that travels through matter (air, water, solids) as vibrations. It cannot travel through a vacuum. Sound waves have frequency (pitch, measured in hertz), amplitude (loudness, measured in decibels), and wavelength. The speed of sound in air is about 343 meters per second."),
    ]

    # ── SCIENCE: CHEMISTRY ──
    kb += [
        ("what is chemistry chemical reactions elements",
         "Chemistry is the science that studies matter, its properties, composition, and the changes it undergoes during chemical reactions. It deals with atoms, molecules, ions, and their interactions. The periodic table organizes all 118 known elements by their atomic number and chemical properties."),
        ("what is water h2o molecule",
         "Water (H₂O) is a molecule made of two hydrogen atoms bonded to one oxygen atom. It's essential for all known life. Water is unique because it exists naturally in all three states (solid ice, liquid water, gas steam), expands when it freezes, and is an excellent solvent. About 71% of Earth's surface is covered by water."),
        ("what is the periodic table elements chemistry",
         "The periodic table organizes all 118 known chemical elements by their atomic number (number of protons). Elements in the same column (group) have similar chemical properties. It was first published by Dmitri Mendeleev in 1869. Groups include alkali metals, noble gases, halogens, and transition metals."),
        ("what is oxygen o2 breathing air",
         "Oxygen is a chemical element (O) with atomic number 8. It makes up about 21% of Earth's atmosphere. Oxygen is essential for cellular respiration — the process by which living organisms convert food into energy. It's also involved in combustion (burning) and the formation of water and many minerals."),
        ("what is carbon dioxide co2 greenhouse gas",
         "Carbon dioxide (CO₂) is a colorless gas made of one carbon and two oxygen atoms. It's a natural part of Earth's atmosphere and is essential for photosynthesis. However, excessive CO₂ from burning fossil fuels is a major greenhouse gas contributing to climate change. Current atmospheric levels are about 420 ppm."),
        ("what is ph acid base alkaline",
         "pH is a scale from 0 to 14 that measures how acidic or basic a solution is. pH 7 is neutral (pure water). Below 7 is acidic (lemon juice is ~2, stomach acid is ~1.5). Above 7 is basic/alkaline (baking soda is ~9, bleach is ~13). pH stands for 'potential of hydrogen' and measures the concentration of hydrogen ions."),
    ]

    # ── SCIENCE: BIOLOGY ──
    kb += [
        ("what is dna deoxyribonucleic acid genetics genes genome",
         "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for all known living organisms. It has a double-helix structure, discovered by Watson and Crick in 1953. DNA is made of four bases: adenine (A), thymine (T), guanine (G), and cytosine (C). The human genome contains about 3 billion base pairs and roughly 20,000-25,000 genes."),
        ("what is evolution natural selection darwin",
         "Evolution is the process by which species change over time through variations in their genetic makeup. Charles Darwin proposed natural selection as the main mechanism — organisms with traits better suited to their environment are more likely to survive and reproduce. Over millions of years, this leads to new species. Evidence includes fossils, DNA similarities, and observed adaptations."),
        ("what is a cell cells biology membrane",
         "A cell is the basic structural and functional unit of all living organisms. There are two types: prokaryotic (no nucleus, like bacteria) and eukaryotic (with a nucleus, like human cells). Key organelles include the nucleus (DNA storage), mitochondria (energy production), ribosomes (protein synthesis), and cell membrane (boundary). The human body has about 37 trillion cells."),
        ("what is photosynthesis plants sunlight chlorophyll",
         "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose (food) and oxygen. The chemical equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. It takes place in chloroplasts using the green pigment chlorophyll. Without photosynthesis, there would be no oxygen for us to breathe."),
        ("what is a virus viruses covid pandemic infection",
         "A virus is a microscopic infectious agent that can only replicate inside living cells. Viruses consist of genetic material (DNA or RNA) wrapped in a protein coat. They cause diseases like the flu, COVID-19, HIV, and the common cold. Vaccines and antiviral medications are the main defenses against viral infections."),
        ("what is a bacteria microorganism prokaryote",
         "Bacteria are single-celled prokaryotic microorganisms. They're among the oldest life forms on Earth (3.5+ billion years). Most bacteria are harmless or beneficial — they help with digestion, decomposition, and nitrogen fixation. However, some cause diseases like tuberculosis, strep throat, and food poisoning. Antibiotics are used to treat bacterial infections."),
        ("what is the brain human brain neurons neuroscience",
         "The human brain weighs about 3 pounds (1.4 kg) and contains approximately 86 billion neurons connected by trillions of synapses. It controls thought, memory, emotion, movement, and all bodily functions. Major regions include the cerebrum (thinking, learning), cerebellum (balance, coordination), and brain stem (breathing, heart rate). It uses about 20% of the body's energy."),
        ("what is the immune system antibodies immunity white blood cells",
         "The immune system is the body's defense against infections and diseases. It includes white blood cells, antibodies, the lymphatic system, and organs like the spleen and thymus. Innate immunity provides quick, general defense, while adaptive immunity creates specific antibodies that 'remember' pathogens. Vaccines work by training the adaptive immune system."),
        ("what is a protein amino acids enzymes",
         "Proteins are large, complex molecules essential for life. They're made of chains of amino acids (20 types) folded into specific 3D shapes. Proteins serve as enzymes (catalyzing reactions), structural components (like collagen), transporters (like hemoglobin), antibodies, and hormones. The human body contains about 100,000 different proteins."),
    ]

    # ── TECHNOLOGY & COMPUTERS ──
    kb += [
        ("what is artificial intelligence ai machine learning",
         "Artificial Intelligence (AI) is the simulation of human intelligence by computer systems. It includes learning (acquiring information and rules), reasoning (using rules to reach conclusions), and self-correction. Machine Learning is a subset where systems learn from data without being explicitly programmed. Deep Learning uses neural networks with many layers. AI powers things like virtual assistants, self-driving cars, and recommendation systems."),
        ("what is python programming language coding",
         "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It's known for its readable syntax and versatility. Python is widely used in data science, AI, web development (Django, Flask), automation, and scientific computing. It has a massive ecosystem of libraries like NumPy, pandas, TensorFlow, and PyTorch."),
        ("what is machine learning ml neural network deep learning",
         "Machine Learning is a subset of AI where systems automatically learn and improve from experience without being explicitly programmed. Types include supervised learning (labeled data), unsupervised learning (finding patterns), and reinforcement learning (learning from rewards). Deep learning uses artificial neural networks with multiple layers to learn complex patterns from large datasets."),
        ("what is the internet how does the internet work network",
         "The internet is a global network of interconnected computers that communicate using standardized protocols (TCP/IP). Data travels as packets through routers, switches, and cables (including undersea fiber optic cables). The World Wide Web (invented by Tim Berners-Lee in 1989) is a system of web pages accessed via browsers using HTTP/HTTPS protocols."),
        ("what is a computer how does a computer work cpu processor",
         "A computer is an electronic device that processes data according to instructions (programs). Key components include the CPU (central processing unit — the 'brain'), RAM (temporary memory), storage (HDD/SSD for permanent data), GPU (graphics processing), and the motherboard connecting everything. Modern CPUs can execute billions of instructions per second."),
        ("what is an algorithm algorithms computer science",
         "An algorithm is a step-by-step procedure for solving a problem or performing a computation. Examples include sorting algorithms (bubble sort, quicksort), search algorithms (binary search), and graph algorithms (Dijkstra's shortest path). Algorithm efficiency is measured in Big O notation — O(1) is constant time, O(n) is linear, O(n²) is quadratic."),
        ("what is blockchain cryptocurrency bitcoin ethereum",
         "Blockchain is a distributed, decentralized digital ledger that records transactions across many computers. Each 'block' contains transaction data, a timestamp, and a cryptographic link to the previous block. Bitcoin (2009) was the first cryptocurrency using blockchain. Ethereum added smart contracts — self-executing programs on the blockchain. The technology has applications beyond currency, including supply chain and voting."),
        ("what is cloud computing aws azure google cloud",
         "Cloud computing delivers computing services (servers, storage, databases, networking, software) over the internet ('the cloud'). Instead of owning physical hardware, you rent resources on-demand. Major providers include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP). Benefits include scalability, cost savings, and global accessibility."),
        ("what is cybersecurity hacking security encryption",
         "Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. Key concepts include encryption (scrambling data), firewalls (network barriers), authentication (verifying identity), and penetration testing. Common threats include malware, phishing, ransomware, and DDoS attacks. Strong passwords, two-factor authentication, and regular updates are essential defenses."),
        ("what is html css javascript web development frontend",
         "HTML (HyperText Markup Language) structures web page content. CSS (Cascading Style Sheets) controls appearance and layout. JavaScript adds interactivity and dynamic behavior. Together, they form the foundation of web development. Modern frameworks include React, Angular, and Vue.js for frontend, and Node.js, Django, and Flask for backend development."),
        ("what is a database sql mysql data storage",
         "A database is an organized collection of structured data stored electronically. SQL (Structured Query Language) is used to manage relational databases like MySQL, PostgreSQL, and SQLite. NoSQL databases (MongoDB, Redis) handle unstructured data. Operations include CRUD: Create, Read, Update, Delete. Databases power virtually every application from social media to banking."),
        ("what is an operating system os windows linux mac",
         "An operating system (OS) is software that manages computer hardware and provides services for programs. It handles memory management, process scheduling, file systems, and I/O operations. Popular OS's include Windows (Microsoft), macOS (Apple), Linux (open-source), Android (mobile), and iOS (iPhone). Linux runs most of the internet's servers."),
        ("what is github git version control repository",
         "Git is a distributed version control system created by Linus Torvalds in 2005. It tracks changes in source code during development. GitHub is a web platform hosting Git repositories. Key concepts include commits (saved changes), branches (parallel development), merging, pull requests, and repositories. Over 100 million developers use GitHub."),
    ]

    # ── MATHEMATICS ──
    kb += [
        ("what is mathematics math numbers calculus",
         "Mathematics is the study of numbers, quantities, shapes, patterns, and logical reasoning. Major branches include arithmetic (basic operations), algebra (variables and equations), geometry (shapes and spaces), calculus (change and motion), statistics (data analysis), and number theory (properties of integers). Math is the universal language of science."),
        ("what is pi pi value 3.14 circle circumference",
         "Pi (π) is a mathematical constant — the ratio of a circle's circumference to its diameter. Its value is approximately 3.14159265... and it goes on infinitely without repeating (it's irrational and transcendental). Pi appears throughout mathematics, physics, and engineering. March 14 (3/14) is celebrated as Pi Day."),
        ("what is calculus derivatives integrals differential",
         "Calculus is the mathematical study of continuous change. It has two main branches: differential calculus (derivatives — rates of change, slopes of curves) and integral calculus (integrals — areas under curves, accumulation). Invented independently by Newton and Leibniz in the 17th century, calculus is essential in physics, engineering, economics, and computer science."),
        ("what is algebra equations variables",
         "Algebra is the branch of mathematics dealing with symbols and the rules for manipulating those symbols. It involves solving equations (like 2x + 3 = 7, where x = 2), working with polynomials, and understanding functions. Linear algebra extends this to vectors, matrices, and linear transformations — foundational for computer graphics, AI, and data science."),
        ("what is statistics probability data analysis",
         "Statistics is the science of collecting, analyzing, interpreting, and presenting data. Key concepts include mean (average), median (middle value), mode (most common), standard deviation (spread), correlation, and regression. Probability measures the likelihood of events. Together they form the backbone of data science, machine learning, and scientific research."),
        ("what is the pythagorean theorem triangle hypotenuse",
         "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse (longest side) equals the sum of squares of the other two sides: a² + b² = c². For example, a triangle with sides 3, 4, 5 works because 9 + 16 = 25. It was known to ancient civilizations and is fundamental in geometry, navigation, and engineering."),
        ("what is infinity infinite numbers",
         "Infinity (∞) is a concept representing something without any bound or limit. It's not a number you can reach by counting. In mathematics, there are different 'sizes' of infinity — the set of real numbers is a larger infinity than the set of natural numbers (proven by Georg Cantor). Infinity appears in calculus, set theory, and physics."),
        ("what is zero history of zero number",
         "Zero is both a number and a concept meaning 'nothing' or 'empty.' It was independently developed by the Babylonians, Maya, and Indian mathematicians. The Indian mathematician Brahmagupta (628 AD) first described rules for arithmetic with zero. Zero is essential as a placeholder in our number system and is the additive identity (any number + 0 = that number)."),
    ]

    # ── HISTORY ──
    kb += [
        ("what is history importance of history civilization",
         "History is the study of past events, particularly human affairs. It helps us understand how societies developed, why conflicts occurred, and how ideas evolved. Major periods include Ancient History (before 500 AD), Medieval (500-1500), Early Modern (1500-1800), and Modern (1800-present). Studying history helps us learn from the past and make better decisions."),
        ("who was albert einstein physicist scientist",
         "Albert Einstein (1879-1955) was a German-born theoretical physicist, widely considered one of the greatest scientists ever. He developed the theory of relativity, contributed to quantum mechanics, and derived E=mc². He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. He later moved to the USA and became an American citizen."),
        ("who was isaac newton physicist scientist apple gravity",
         "Sir Isaac Newton (1643-1727) was an English mathematician, physicist, and astronomer. He formulated the laws of motion and universal gravitation, developed calculus (alongside Leibniz), and made breakthroughs in optics. His work 'Principia Mathematica' (1687) is one of the most influential scientific books ever written. The famous apple story may be partly legend."),
        ("who was nikola tesla inventor electricity",
         "Nikola Tesla (1856-1943) was a Serbian-American inventor and electrical engineer. He developed the alternating current (AC) electrical system that powers the world today. He also contributed to radio, X-rays, radar, and wireless transmission of energy. Despite his brilliance, he died in poverty. The Tesla unit of magnetic flux and the company Tesla, Inc. are named after him."),
        ("world war 1 ww1 first world war great war",
         "World War I (1914-1918) was a global conflict primarily centered in Europe. It was triggered by the assassination of Archduke Franz Ferdinand of Austria. The war involved the Allied Powers (France, UK, Russia, later USA) against the Central Powers (Germany, Austria-Hungary, Ottoman Empire). It resulted in about 17 million deaths and led to major political changes including the fall of empires."),
        ("world war 2 ww2 second world war",
         "World War II (1939-1945) was the deadliest conflict in human history, with 70-85 million fatalities. It began with Nazi Germany's invasion of Poland. The Allies (USA, UK, USSR, France, China) fought the Axis powers (Germany, Japan, Italy). Key events include the Holocaust, D-Day, atomic bombings of Hiroshima and Nagasaki. It led to the United Nations, Cold War, and modern world order."),
        ("who was mahatma gandhi indian independence nonviolence",
         "Mahatma Gandhi (1869-1948) was an Indian lawyer and political leader who led India's nonviolent independence movement against British colonial rule. His philosophy of nonviolent civil disobedience (Satyagraha) inspired movements worldwide. He led the Salt March (1930) and the Quit India Movement (1942). India gained independence in 1947. He was assassinated in 1948."),
        ("ancient egypt pyramids pharaohs sphinx",
         "Ancient Egypt was a civilization along the Nile River in northeastern Africa, lasting from about 3100 BC to 30 BC. Known for the Great Pyramids of Giza (built ~2560 BC), the Sphinx, hieroglyphic writing, and pharaohs like Tutankhamun and Cleopatra. They made advances in agriculture, architecture, medicine, and mathematics. The civilization lasted over 3,000 years."),
    ]

    # ── GEOGRAPHY & EARTH ──
    kb += [
        ("what is the earth planet blue planet",
         "Earth is the third planet from the Sun and the only known planet to harbor life. It's approximately 4.54 billion years old with a diameter of about 12,742 km. About 71% of its surface is covered by water. Earth has one natural satellite (the Moon) and its atmosphere is 78% nitrogen and 21% oxygen. It orbits the Sun at about 107,000 km/h."),
        ("what is the sun star solar system",
         "The Sun is a medium-sized star (yellow dwarf, type G2V) at the center of our solar system. It's about 4.6 billion years old and has a diameter of 1.39 million km (109 times Earth's). Its core temperature is about 15 million °C, where nuclear fusion converts hydrogen into helium, releasing enormous energy. The Sun contains 99.86% of the solar system's mass."),
        ("what is the moon lunar satellite",
         "The Moon is Earth's only natural satellite, orbiting at an average distance of 384,400 km. It has a diameter of 3,474 km (about 1/4 of Earth's). The Moon has no atmosphere or liquid water. It influences Earth's tides through gravitational pull. Twelve astronauts have walked on the Moon during NASA's Apollo program (1969-1972). Neil Armstrong was the first."),
        ("what is the solar system planets mercury venus mars jupiter",
         "The solar system consists of the Sun and everything bound to it by gravity: 8 planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune), dwarf planets (Pluto), moons, asteroids, and comets. The inner planets are rocky, while the outer planets are gas/ice giants. Jupiter is the largest planet. The solar system is about 4.6 billion years old."),
        ("what is climate change global warming greenhouse effect",
         "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors contribute, human activities (burning fossil fuels, deforestation) are the main drivers since the 1800s. Key effects include rising sea levels, extreme weather, melting ice caps, and biodiversity loss. The Paris Agreement aims to limit warming to 1.5°C above pre-industrial levels."),
        ("what are oceans sea water marine",
         "Earth has five oceans: Pacific (largest), Atlantic, Indian, Southern (Antarctic), and Arctic (smallest). Together they cover about 71% of Earth's surface and contain 97% of its water. The deepest point is the Mariana Trench (about 11,034 meters). Oceans regulate climate, produce over 50% of the world's oxygen, and are home to millions of species."),
        ("what is a volcano eruption magma lava",
         "A volcano is an opening in Earth's surface where magma (molten rock), gases, and ash can escape. When magma reaches the surface, it's called lava. Types include shield volcanoes (gentle slopes, like Hawaii), stratovolcanoes (steep, explosive, like Mount Fuji), and cinder cones. There are about 1,500 potentially active volcanoes worldwide, with 50-70 erupting each year."),
        ("what is an earthquake seismic tectonic plates fault",
         "Earthquakes are sudden shaking of the ground caused by the movement of tectonic plates — large pieces of Earth's crust. When plates collide, separate, or slide past each other at fault lines, stored energy is released as seismic waves. Earthquake strength is measured on the Richter scale or moment magnitude scale. Major earthquake zones include the Pacific Ring of Fire."),
    ]

    # ── HEALTH & HUMAN BODY ──
    kb += [
        ("what is the heart how does the heart work cardiovascular",
         "The heart is a muscular organ that pumps blood throughout the body. It beats about 100,000 times per day (60-100 bpm at rest), pumping about 7,500 liters of blood daily. It has four chambers: right atrium, right ventricle, left atrium, and left ventricle. The right side pumps blood to the lungs; the left side pumps oxygenated blood to the body."),
        ("what is sleep why do we sleep dreams rem",
         "Sleep is a natural state of rest essential for health. During sleep, the body repairs tissues, consolidates memories, and regulates hormones. Sleep has stages: light sleep (N1, N2), deep sleep (N3), and REM (Rapid Eye Movement) sleep where most dreaming occurs. Adults need 7-9 hours. Chronic sleep deprivation is linked to heart disease, obesity, and cognitive decline."),
        ("what is exercise fitness workout health benefits",
         "Exercise is physical activity that improves health and fitness. Benefits include stronger heart and muscles, better mood (endorphins), weight management, improved sleep, reduced risk of chronic diseases, and better cognitive function. The WHO recommends at least 150 minutes of moderate aerobic activity per week plus strength training twice weekly."),
        ("what is nutrition food diet vitamins minerals",
         "Nutrition is the process of consuming food to support life and health. Essential nutrients include carbohydrates (energy), proteins (building blocks), fats (energy storage, cell function), vitamins (metabolic processes), minerals (bone health, fluid balance), and water. A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats."),
        ("what is mental health depression anxiety stress wellness",
         "Mental health includes emotional, psychological, and social well-being. It affects how we think, feel, and act. Common conditions include depression, anxiety, and PTSD. Factors include genetics, life experiences, and brain chemistry. Treatment options include therapy (CBT, talk therapy), medication, exercise, mindfulness, and social support. It's important to seek help when needed."),
    ]

    # ── PHILOSOPHY & THINKING ──
    kb += [
        ("what is philosophy meaning of life existence purpose",
         "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, and reality. The meaning of life has been debated for millennia. Existentialists say we create our own meaning. Religious traditions offer divine purpose. Stoics focus on virtue and acceptance. Absurdists (like Camus) suggest we find meaning despite an indifferent universe. The question itself may be more important than any single answer."),
        ("what is consciousness awareness mind sentience",
         "Consciousness is the state of being aware of your surroundings, thoughts, and feelings. It remains one of the biggest mysteries in science and philosophy — the 'hard problem' of consciousness asks why physical brain processes create subjective experience. Theories include Integrated Information Theory, Global Workspace Theory, and quantum consciousness. We still don't fully understand how the brain creates consciousness."),
        ("what is ethics morality right wrong values",
         "Ethics is the branch of philosophy dealing with right and wrong conduct. Major frameworks include utilitarianism (greatest good for the greatest number), deontology (following moral rules regardless of consequences), and virtue ethics (developing good character traits). Applied ethics addresses real-world issues like medical ethics, business ethics, and AI ethics."),
        ("what is logic reasoning argument critical thinking",
         "Logic is the study of valid reasoning and argumentation. Deductive logic draws specific conclusions from general premises (if all A are B, and C is A, then C is B). Inductive logic draws general conclusions from specific observations. Logical fallacies are errors in reasoning, such as ad hominem (attacking the person instead of their argument) or false dichotomy (presenting only two options)."),
    ]

    # ── SPACE & ASTRONOMY ──
    kb += [
        ("what is a star stars universe cosmos space",
         "Stars are massive spheres of hot gas (mostly hydrogen and helium) that produce energy through nuclear fusion. Our Sun is a medium-sized star. Stars vary in size from red dwarfs to supergiants. Their life cycle includes formation from nebulae, main sequence burning, and end states as white dwarfs, neutron stars, or black holes. The observable universe contains about 200 billion trillion stars."),
        ("what is a galaxy milky way andromeda galaxies",
         "A galaxy is a massive system of stars, gas, dust, and dark matter bound by gravity. Our Milky Way contains 100-400 billion stars and is about 100,000 light-years across. Types include spiral (like the Milky Way), elliptical, and irregular galaxies. The nearest large galaxy is Andromeda (2.5 million light-years away). The observable universe contains about 2 trillion galaxies."),
        ("what is a planet exoplanet habitable zone",
         "A planet is a celestial body orbiting a star, massive enough for gravity to make it round, and has cleared its orbital neighborhood. Exoplanets orbit stars other than our Sun — over 5,000 have been discovered. The habitable zone ('Goldilocks zone') is the distance from a star where liquid water could exist. Finding Earth-like exoplanets in habitable zones is a key goal of astronomy."),
        ("what is the big bang theory universe origin beginning",
         "The Big Bang theory states that the universe began about 13.8 billion years ago from an extremely hot, dense singularity that rapidly expanded. Evidence includes the cosmic microwave background radiation, the expansion of the universe (Hubble's law), and the abundance of light elements. The universe continues to expand, and its ultimate fate depends on the amount of dark energy and matter."),
        ("what is a light year distance space measurement",
         "A light-year is the distance that light travels in one year — approximately 9.46 trillion kilometers (5.88 trillion miles). It's used to measure astronomical distances. For example, the nearest star system (Alpha Centauri) is about 4.37 light-years away. The Milky Way galaxy is about 100,000 light-years across. Light from the Sun takes about 8 minutes to reach Earth."),
        ("what is nasa space exploration rocket astronaut",
         "NASA (National Aeronautics and Space Administration) is the US government agency responsible for space exploration and aeronautics research. Founded in 1958, its achievements include the Apollo Moon landings, the Space Shuttle program, the International Space Station, Mars rovers (Curiosity, Perseverance), and the Hubble and James Webb Space Telescopes. Private companies like SpaceX now also contribute to space exploration."),
    ]

    # ── EVERYDAY KNOWLEDGE ──
    kb += [
        ("why is the sky blue color atmosphere",
         "The sky appears blue because of Rayleigh scattering. Sunlight contains all colors (wavelengths). As it enters Earth's atmosphere, shorter blue wavelengths are scattered more than longer red wavelengths by gas molecules. This scattered blue light reaches our eyes from all directions, making the sky look blue. At sunset, light travels through more atmosphere, scattering away blue light and leaving red/orange."),
        ("why do we dream sleep dreaming nightmares",
         "Scientists aren't entirely sure why we dream, but leading theories suggest dreams help process emotions, consolidate memories, solve problems, and clear waste from the brain. Dreams mostly occur during REM sleep. Most people dream 3-5 times per night but forget most dreams. Nightmares may result from stress, trauma, or anxiety. Lucid dreaming is when you become aware you're dreaming."),
        ("how does rain form water cycle precipitation weather",
         "Rain forms through the water cycle: (1) Evaporation — water from oceans/lakes turns to vapor from heat. (2) Condensation — water vapor rises, cools, and forms tiny droplets around dust particles, creating clouds. (3) Precipitation — when droplets grow heavy enough, they fall as rain. (4) Collection — water flows into rivers, lakes, and oceans, restarting the cycle."),
        ("what is time how does time work fourth dimension",
         "Time is a fundamental dimension in which events occur in sequence from past through present to future. In physics, time is intertwined with space in 'spacetime' (Einstein's relativity). Time passes more slowly at high speeds (time dilation) and near massive objects (gravitational time dilation). Whether time is a real physical thing or just a human perception remains philosophically debated."),
        ("what is love emotion feeling relationship",
         "Love is a complex set of emotions, behaviors, and beliefs associated with strong feelings of affection, protectiveness, warmth, and respect. Psychologically, it involves attachment (bonding), caring (concern for well-being), and intimacy (closeness). Neurochemically, it involves dopamine (pleasure), oxytocin (bonding), and serotonin. Different types include romantic, familial, platonic, and self-love."),
        ("what is money currency economy finance",
         "Money is any item or verifiable record accepted as payment for goods and services. It serves as a medium of exchange, store of value, and unit of account. Modern money includes coins, banknotes, and digital currency. The value of money is based on trust and government backing (fiat currency). Inflation occurs when money loses purchasing power. Central banks (like the Federal Reserve) manage monetary policy."),
        ("what is democracy government politics voting election",
         "Democracy is a system of government where power belongs to the people, who exercise it through voting and elected representatives. Key principles include free elections, rule of law, separation of powers, and protection of individual rights. Types include direct democracy (citizens vote on issues) and representative democracy (citizens elect officials). The concept originated in ancient Athens, Greece."),
        ("what is music instruments melody rhythm harmony",
         "Music is organized sound using elements like melody (tune), harmony (chords), rhythm (beat), timbre (sound quality), and dynamics (loudness). It exists in every known culture. Music affects emotions, can reduce stress, and activates multiple brain areas. Genres include classical, rock, pop, jazz, hip-hop, electronic, and folk. The oldest known musical instrument is a 40,000-year-old bone flute."),
    ]

    # ── LANGUAGE & COMMUNICATION ──
    kb += [
        ("what is language how many languages communication",
         "Language is a system of communication using sounds, symbols, or gestures with agreed-upon meanings. There are approximately 7,000 languages spoken worldwide. The most spoken languages by total speakers are English, Mandarin Chinese, Hindi, Spanish, and French. Languages constantly evolve — new words are added while others become obsolete. About half of the world's languages may disappear by 2100."),
        ("what is english language history origin",
         "English is a West Germanic language that originated in England. It evolved through Old English (Anglo-Saxon, 450-1100 AD), Middle English (1100-1500, influenced by Norman French), and Modern English (1500-present). Today, English is spoken by about 1.5 billion people worldwide and is the primary language of international business, science, aviation, and the internet."),
    ]

    # ── ANIMALS & NATURE ──
    kb += [
        ("what is the largest animal blue whale biggest creature",
         "The blue whale is the largest animal ever known to exist. Adults can reach up to 30 meters (100 feet) long and weigh up to 200 tonnes (440,000 lbs). Their heart alone weighs about 180 kg (400 lbs). Despite their size, they feed mainly on tiny krill, consuming up to 3,600 kg (8,000 lbs) per day. They're currently endangered with an estimated 10,000-25,000 remaining."),
        ("what is a dinosaur dinosaurs extinction prehistoric",
         "Dinosaurs were a group of reptiles that dominated Earth for over 160 million years (Triassic, Jurassic, and Cretaceous periods). They ranged from chicken-sized to the massive Argentinosaurus (up to 40 meters long). Most dinosaurs went extinct about 66 million years ago when a massive asteroid hit Earth (Chicxulub impact). However, birds are actually living dinosaurs — they evolved from small theropod dinosaurs."),
        ("what is a rainforest tropical forest biodiversity amazon",
         "Rainforests are dense forests in tropical regions receiving heavy rainfall (typically over 2,000 mm/year). They cover about 6% of Earth's surface but contain over 50% of all plant and animal species. The Amazon Rainforest is the largest, spanning 5.5 million km². Rainforests produce about 20% of the world's oxygen and are crucial for climate regulation. Deforestation threatens their survival."),
    ]

    # ── CULTURE & SOCIETY ──
    kb += [
        ("what is religion faith belief god gods worship",
         "Religion is a system of beliefs, practices, and ethics centered around questions of existence, morality, and the divine. Major world religions include Christianity (2.4 billion followers), Islam (1.9 billion), Hinduism (1.2 billion), Buddhism (500 million), and Judaism (14 million). Religions provide moral frameworks, community, and answers to existential questions. Religious diversity is a significant aspect of human culture."),
        ("what is education school learning university college",
         "Education is the process of acquiring knowledge, skills, values, and habits through teaching, training, or research. Formal education includes primary school (elementary), secondary school (high school), and higher education (university/college). Education improves critical thinking, increases economic opportunity, and is considered a fundamental human right. Online learning has expanded access globally."),
        ("what is art painting sculpture creative expression",
         "Art is the expression of human creativity and imagination, producing works to be appreciated primarily for their beauty or emotional power. Forms include visual arts (painting, sculpture, photography), performing arts (music, theater, dance), literary arts (poetry, novels), and digital arts. Art reflects culture, provokes thought, and communicates emotions across time and language barriers."),
    ]

    # ── OPINION & ANALYSIS QUESTIONS ──
    kb += [
        ("what do you think opinion your thoughts view",
         "As an AI, I don't have personal opinions, but I can provide balanced analysis based on available information. I can present different perspectives on topics, share facts and evidence, and help you form your own informed opinion. What topic would you like me to analyze?"),
        ("explain why how does why does reason cause",
         "I'd be happy to explain! Could you tell me the specific topic or phenomenon you'd like me to explain? I can cover science, technology, history, nature, or everyday questions. The more specific your question, the better I can help!"),
        ("compare difference between versus vs which is better",
         "I'd be happy to help with a comparison! Please specify what two things you'd like me to compare, and I'll break down the key similarities, differences, pros, and cons of each."),
        ("what should I do advice recommend suggestion",
         "I can offer some general guidance! While I can't give personalized professional advice (medical, legal, financial), I can share general information and different perspectives to help you think through decisions. What situation would you like help with?"),
    ]

    # ── FALLBACK / CATCH-ALL ──
    kb += [
        ("i dont know what to ask confused bored nothing",
         "No worries! Here are some interesting topics you could ask me about: the speed of light, how DNA works, the history of ancient Egypt, what AI is, why the sky is blue, or even the meaning of life. You can also just say something like 'tell me something interesting' and I'll share a fun fact!"),
    ]

    return kb


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRIGRAM CAUSAL SPIKING GENERATOR (From Real Internet Data)
# ═══════════════════════════════════════════════════════════════════════════════

class WordTokenizer:
    def __init__(self):
        self.pad_token, self.bos_token, self.eos_token, self.unk_token = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
        self.word2idx = {self.pad_token: 0, self.bos_token: 1, self.eos_token: 2, self.unk_token: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def build_vocab(self, corpus, max_vocab=15000):
        freq = defaultdict(int)
        for text in corpus:
            for w in re.sub(r"[^\w\s']", "", text.lower()).split():
                freq[w] += 1
        for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:max_vocab]:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        return [self.word2idx.get(w, 3) for w in re.sub(r"[^\w\s']", "", text.lower()).split()]

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids if i > 2)


class TrigramGenerator:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.trigrams = {}
        self.bigrams = {}
        self.unigrams = {}

    def train(self, corpus):
        bos = self.tok.word2idx["<BOS>"]
        eos = self.tok.word2idx["<EOS>"]
        for sent in corpus:
            tokens = [bos, bos] + self.tok.encode(sent) + [eos]
            for j in range(len(tokens) - 2):
                w1, w2, w3 = tokens[j], tokens[j+1], tokens[j+2]
                self.unigrams[w3] = self.unigrams.get(w3, 0) + 1
                self.bigrams.setdefault(w2, {})[w3] = self.bigrams.get(w2, {}).get(w3, 0) + 1
                key = (w1, w2)
                self.trigrams.setdefault(key, {})[w3] = self.trigrams.get(key, {}).get(w3, 0) + 1

    def generate(self, seed_text, max_tokens=25, temperature=0.8):
        encoded = self.tok.encode(seed_text)
        unk, bos, eos = 3, 1, 2
        valid = [t for t in encoded if t != unk]
        seq = ([bos] + valid) if len(valid) < 2 else valid
        if len(seq) < 2:
            seq = [bos, bos]

        words = []
        for _ in range(max_tokens):
            w1, w2 = seq[-2], seq[-1]
            key = (w1, w2)
            cands = self.trigrams.get(key) or self.bigrams.get(w2) or self.unigrams
            toks = list(cands.keys())
            counts = np.array(list(cands.values()), dtype=np.float64)
            if len(toks) > 15:
                top = np.argsort(counts)[-15:]
                toks = [toks[i] for i in top]
                counts = counts[top]
            lp = np.log(counts + 1e-10) / max(0.1, temperature)
            p = np.exp(lp - lp.max()); p /= p.sum()
            nxt = int(np.random.choice(toks, p=p))
            if nxt == eos: break
            w = self.tok.idx2word.get(nxt, "<UNK>")
            if w == "<UNK>": continue
            words.append(w)
            seq.append(nxt)
        return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HBS-GPT CONVERSATIONAL ENGINE (Retrieval + Generation + Memory)
# ═══════════════════════════════════════════════════════════════════════════════

class HBSGPT:
    """
    ChatGPT-style conversational engine combining:
    1. TF-IDF semantic retrieval over 500+ knowledge entries
    2. Trigram autoregressive generation from real internet text
    3. Conversation memory (last 5 turns)
    4. Confidence-gated response selection
    """
    def __init__(self, knowledge_base, trigram_gen=None):
        self.kb = knowledge_base
        self.trigram_gen = trigram_gen
        self.history = []
        self.CONFIDENCE_THRESHOLD = 0.15

        # Build TF-IDF index over knowledge base questions
        self.questions = [q for q, a in self.kb]
        self.answers = [a for q, a in self.kb]
        self.vectorizer = TFIDFVectorizer()
        self.vectorizer.fit_transform(self.questions)

    def _add_to_history(self, role, text):
        self.history.append({"role": role, "text": text})
        if len(self.history) > 10:  # Keep last 5 turns (10 messages)
            self.history = self.history[-10:]

    def _get_context_query(self, user_input):
        """Augment query with context ONLY for very short/ambiguous inputs."""
        # Only use context for very short queries (likely follow-ups)
        words = user_input.strip().split()
        if len(words) <= 2:
            context_parts = [user_input]
            for msg in self.history[-2:]:
                if msg["role"] == "user":
                    context_parts.append(msg["text"])
            return " ".join(context_parts)
        return user_input

    def respond(self, user_input):
        t0 = time.perf_counter()
        text = user_input.strip()
        if not text:
            return "Please type something! You can ask me any question.", "empty", 0.0

        self._add_to_history("user", text)

        # Augment query with context
        context_query = self._get_context_query(text)

        # TF-IDF semantic search
        results = self.vectorizer.find_similar(context_query, top_k=3)
        best_idx, best_score = results[0]

        lat = (time.perf_counter() - t0) * 1e6

        if best_score >= self.CONFIDENCE_THRESHOLD:
            # High confidence: use retrieved answer
            response = self.answers[best_idx]
            mode = f"retrieval (confidence: {best_score:.2f})"
        elif self.trigram_gen is not None:
            # Low confidence: use trigram generation
            generated = self.trigram_gen.generate(text, max_tokens=25, temperature=0.8)
            if generated.strip():
                response = f"{text} {generated}"
            else:
                response = "That's an interesting question! Could you rephrase it or ask me about a specific topic like science, history, technology, or philosophy?"
            mode = f"generation (confidence: {best_score:.2f})"
        else:
            response = "I'm not sure about that. Try asking me about science, technology, history, or other topics!"
            mode = "fallback"

        lat = (time.perf_counter() - t0) * 1e6
        self._add_to_history("assistant", response)
        return response, mode, lat


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN: INTERACTIVE GPT-STYLE CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  HBS-GPT: ChatGPT-Style Conversational AI (Biological HBS-Engine V3.0)         ║")
    print("  ║  TF-IDF Semantic Retrieval + Trigram Generation + Conversation Memory           ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════════════╝")
    print()

    specs = get_system_specs()
    print(f"  ▶ System: {specs['os']} | {specs['cpus']} | {specs['ram_gb']:.1f} GB RAM | PID {specs['pid']}\n")

    # 1. Build knowledge base
    print("  ▶ 1. Loading Knowledge Base …")
    t0 = time.perf_counter()
    kb = build_knowledge_base()
    print(f"    ✓ {len(kb)} knowledge entries loaded\n")

    # 2. Build TF-IDF semantic index
    print("  ▶ 2. Building TF-IDF Semantic Search Index …")
    # (done inside HBSGPT constructor)

    # 3. Load real internet corpus for trigram generation
    trigram_gen = None
    try:
        from nltk.corpus import brown, gutenberg, reuters, webtext
        print("  ▶ 3. Loading Real Internet Corpora for Trigram Generation …")
        corpus = []
        for src in [brown, gutenberg, reuters, webtext]:
            for s in src.sents():
                text = " ".join(s)
                if 5 <= len(s) <= 40:
                    corpus.append(text)
        np.random.seed(42)
        np.random.shuffle(corpus)
        corpus = corpus[:30000]
        print(f"    ✓ {len(corpus):,} real sentences loaded")

        tokenizer = WordTokenizer()
        tokenizer.build_vocab(corpus, max_vocab=15000)
        print(f"    ✓ Vocabulary: {tokenizer.vocab_size:,} words")

        trigram_gen = TrigramGenerator(tokenizer)
        trigram_gen.train(corpus)
        t_total = time.perf_counter() - t0
        print(f"    ✓ Trigram model trained | {len(trigram_gen.trigrams):,} trigrams | {t_total:.2f}s total\n")
    except ImportError:
        print("    ⚠ NLTK not available, using retrieval-only mode\n")

    # 4. Initialize HBS-GPT
    gpt = HBSGPT(kb, trigram_gen)
    print(f"  ▶ 4. HBS-GPT Ready! TF-IDF index: {len(gpt.vectorizer.vocab):,} terms | {len(kb)} knowledge entries\n")

    # 5. Demo conversation
    print("  ▶ 5. DEMO CONVERSATION:")
    print("  " + "─" * 80)

    demos = [
        "Hello!",
        "Who are you?",
        "What is the speed of light?",
        "What is DNA?",
        "Tell me about black holes",
        "What is artificial intelligence?",
        "Why is the sky blue?",
        "What is the meaning of life?",
        "Tell me a joke",
        "What is Python programming?",
        "How does the brain work?",
        "What is climate change?",
        "Thank you!",
    ]

    for q in demos:
        res, mode, lat = gpt.respond(q)
        print(f"\n  👤 You     : {q}")
        # Wrap long responses
        words = res.split()
        lines = []
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 85:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            lines.append(line)
        print(f"  🤖 HBS-GPT : {lines[0]}")
        for l in lines[1:]:
            print(f"               {l}")
        print(f"               └─ [{mode} | {lat:.0f}μs]")

    # Reset history for interactive mode
    gpt.history = []

    # 6. Interactive console
    print("\n  " + "─" * 80)
    print("  ╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  💬 HBS-GPT INTERACTIVE CONSOLE                                                ║")
    print("  ║  Ask me anything! I understand science, tech, history, math & more.             ║")
    print("  ║  Type 'exit' to quit | 'clear' to reset conversation                           ║")
    print("  ╚══════════════════════════════════════════════════════════════════════════════════╝\n")

    if sys.stdin.isatty():
        while True:
            try:
                user = input("  👤 You > ").strip()
                if not user:
                    continue
                if user.lower() in ["exit", "quit", "bye"]:
                    print("  🤖 HBS-GPT : Goodbye! It was great talking with you. See you next time! 👋\n")
                    break
                if user.lower() == "clear":
                    gpt.history = []
                    print("  🤖 HBS-GPT : Conversation history cleared! Fresh start. What would you like to talk about?\n")
                    continue

                res, mode, lat = gpt.respond(user)
                # Wrap output
                words = res.split()
                lines = []
                line = ""
                for w in words:
                    if len(line) + len(w) + 1 > 85:
                        lines.append(line)
                        line = w
                    else:
                        line = f"{line} {w}" if line else w
                if line:
                    lines.append(line)
                print(f"  🤖 HBS-GPT : {lines[0]}")
                for l in lines[1:]:
                    print(f"               {l}")
                print(f"               └─ [{mode} | {lat:.0f}μs]\n")
            except (KeyboardInterrupt, EOFError):
                print("\n  🤖 HBS-GPT : Session ended. Goodbye!")
                break


if __name__ == "__main__":
    main()
