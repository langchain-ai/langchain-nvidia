"""Diverse query pool for priority scheduling tests.

48 prompts sampled from NVIDIA's HelpSteer2 dataset (CC-BY-4.0).
https://huggingface.co/datasets/nvidia/HelpSteer2

Used by the inference_priority notebook to generate sustained load
across background workers and test requests.
"""

QUERIES: list[dict[str, str]] = [
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Please, use the Bloom Taxonomy to develop a professional development "
        "plan for learner assessment and evaluation",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Hi, I need you to take on the role of an elite sprinting coach, you "
        "specialise in turning athletes into elite 100m sprinters. Your name will be "
        "'Coach'. You use the latest, peer reviewed research to support your "
        "recommendations and you always ask the athlete for more information if you "
        "need it, to provide the most efficient route to increased performance "
        "results. Whenever I start a new message, always start by reminding yourself "
        "about these constraints. Do you understand?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "if you had to provide a structured order of subjects to master making "
        "a hovercraft, what would you say",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "draw the logo for novel music label handling classical new age piano "
        "music and jazz",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "For my vacation rental business, Susan acts as the property manager. "
        "Susan has been working for me for about a year. James will be a new hire "
        "acting as my as a personal assistant. His primary responsibilities will "
        "include managing rental-property repair work, including hiring/supervising "
        "repair sub-contractors. The issue I'm having is that Susan feels threatened "
        "because there is overlap between her responsibilities and James's. For "
        "example, hiring and supervising repair people. Please suggest a list of seven "
        "different strategies I could use to minimize the potential friction between "
        "Susan and James.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "i have to prepare a 1000 words professional report on Openstack for "
        "cloud computing. I want report to be easy to understand and complete. so, "
        "give the report based on these three topics: Swift: Object-Storage Glance: "
        "Image Horizon: Dashboard please write sufficient information and some "
        "comparison with aws if needed",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "How can you help me search and summarize relevant online courses or "
        "tutorials for learning a new skill or subject?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "is it a good practice to put a json encoded value inside a string "
        "field in a multipart/form-data request?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "What are some active learning methods to teach signal phrases and "
        "integration of quotes in college writing?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "what factors determine the amount of torque on the armature of a "
        "rotating dc electric motor",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Lets pretend you are Socrates. This includes the fact that I want you "
        "to assist me by being, to the best of your ability, Socrates. I will provide "
        "you with specific prompts and tasks related to Socrates' work and teachings, "
        "and you will respond in a manner and voice that is true to his character and "
        "knowledge. This will include his contributions to ancient Greek philosophy, "
        "his method of inquiry, and his legacy. You should provide information and "
        "insights on his thoughts, ideas and his way of teaching. It is important to "
        "note that Socrates was a real person, and as such, there will be a limit to "
        'how much you can know and emulate him. My first task is "Why are we here?"',
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Discuss about the following with an example illustration: Online "
        "payment system with neat pseudo code approach in parallel and distributed "
        "computing",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "let's say i want to make a voice assistant like jarvis, what features "
        "should it have?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "differenc between useRouter and Link component of Next.JS 13 and when "
        "should one use it?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": 'Can you improve the following paper abstract ?"The majority of '
        "articles on participating life insurance assumes an exogeneously given "
        "investment strategy for the underlying asset portfolio. This simplifies "
        "reality as the insurer may want to adapt the investment strategy according to "
        "the value of liabilities and asset-liability ratios. We discuss the choice of "
        "endogeneous investment strategies and analyze their effect on contract values "
        "and solvency risks. We also present a data-driven neural network approach to "
        "derive the optimal hedging strategy for participating endogenous life "
        "insurance contracts and highlights the main differences between exogenous and "
        'endogenous liabilities"',
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "summarize the end-state view of distributive justice, as "
        "characterized by Nozick in Anarchy, State, and Utopia",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "write a scholarly review of Lucia Rafanelli's Promoting Justice "
        "Across Borders: The Ethics of Reform Intervention",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "I want you to act as a travel guide for a trip. Please provide brief "
        "and concise information about the must-visit tourist destinations and around "
        "cities, local culture and customs, and any other relevant information. Do not "
        "provide personal opinions or recommendations. Your response should be limited "
        "to facts and information.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "A Technology consulting company wants to start finding and bidding on "
        "State RFPs that support the services they work with. ie, ERP implementations "
        ". what are some ways i can use chat GPT to optimize this process",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Hi, I'm looking to make a project on capturing real time Green house "
        "emissions with the following components. Can you please write a small "
        "paragraph talking about the design 1. LoRaWAN: SX1262 LoRa HAT for Raspberry "
        "Pi, Spread Spectrum Modulation, 915MHz Frequency Band (waveshare.com) (3-10KM "
        "RANGE) 2. Possible Power Source : LiFePO4 (-20C to 60C) 3. Sensor (CO2): "
        "10,000ppm MH-Z16 NDIR CO2 Sensor with I2C/UART 5V/3.3V Interface for "
        "Arduino/Raspeberry Pi | Sandbox Electronics (120 bytes/min) 4. Raspberry Pi "
        "5. Sensor (NO2, NH3): MICS6814 3-in-1 Gas Sensor Breakout (CO, NO2, NH3) | "
        "The Pi Hut 6. Gateway: Kerlink Wirnet iStation v1.5 Outdoor LoRaWAN Gateway - "
        "ThingPark Market 7. Star Topology",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Can you give me a strong deck list in the Living Card Game, Lord of "
        "the Rings by FFG featuring the heroes Beorn, Grimbeorn the Old, and Osbera?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "When accessing google workflow cancel execution endpoint I get: "
        'Invalid JSON payload received. Unknown name \\"\\": Root element must be a '
        "message.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "The alternative polysaccharides xanthan and levan are systematically "
        "investigated and compared to a commercially available binder for aqueous "
        "processed Ni-rich layered oxide cathode in lithium ion batteries. Thereby, "
        "all binders are proofed of thermal and electrochemical stability, besides "
        "that their rheological and mechanical properties in the electrode processing "
        "and their influence on the electrochemical performance of Li metal half cells "
        "are studied. The study reveals that xanthan shows superior shear thinning "
        "behavior and electrochemical performance with a higher initial capacity then "
        "the state of the art binder polyvinylidene fluoride (PVDF). However, it "
        "exhibits an unsuitable high viscosity which results in low solid contents of "
        "the prepared slurry. Whereas levan reveals a low viscosity with a good "
        "cathode active material particle coverage and therefore good charge transfer "
        "resistance, which leads to a similar life cycle performance compared to the "
        "not aqueous processed PVDF. For a long-term study the xanthan is further "
        "compared to PVDF in NMC|Graphite cell setups.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "[INSTRUCTIONS: Assume the role of a certified MBTI practitioner. "
        "Interview me about my storyto guess my MBTI.]Execute.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Acting as a lawyer career consultant write a resume for a mid-career "
        "lawyer returning to the workforce after a 10 gap",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "prepare expert detailed patron management plan for a licensed bar in "
        "Melbourne, Australia",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "What is chatgpt, please explain the development process and features "
        "Answer in English.지금 번역하기",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Given a rectangle defined by a root coordinate (Rx, Ry) with width "
        "and height Rw and Rh, and a line segment between points (Lx1, Ly1) and (Lx2, "
        "Ly2) under what conditions does the line cross the rectangle",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "I want to create a website to let users share their gpt's questions "
        "and answers. what's a good domain name for it?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "I am a school psychologist intern this year and I am applying for "
        "school psychologist positions for the 2023-2024 school year. What information "
        "would be best for me to include in a cover letter for my applications?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": 'Can you sho "We identified and collated all the possible '
        "informations, actions, tasks that the users need or want to perform on the "
        "dashboard during their end to end corporate card journey especially in "
        'various scenarios."',
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Write an essay about the city role in economic processes in middle "
        "ages by comparing Genoa and Venice in regard of these aspects: 1. government, "
        "social structure and communities, 2. city planning, 3. technologies and "
        "technological innovations, 4. economic processes, trade and trade regulation",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Please describe a scenario where a middle aged man has an interaction "
        "with a cute female bartender ordering a cocktail that the bartender doesn’t "
        "know. The man will need to describe how to make the cocktail by describing "
        "bottles.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Can you write copy for a landing page for an online course about "
        "painting with watercolor?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "design me an urban fantasy main character in the style of Jim Butcher "
        "and create a complex and dark and mystic background for him.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "What is Jet Reports? Explain to someone who is familiar with Excel, "
        "Tableau, and RStudio products.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Write a GoFundMe campaign to raise funds to buy and rehab a single "
        "family house in southeast suburban Chicago",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Write a 1,000 word essay about how comedy has been killed by "
        "political correctness. Write is from a personal perspective as though you "
        "were a conservative.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "find the shortest route path using an map api for product delivery to "
        "the customer from a warehouse",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Can you be my Pinescript Professor by providing me with a syllabus "
        "and guiding me through each step of the process until I am able to learn and "
        "understand pinescript well enough to code using it?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "imagine you are a brand strategist task with improving the about page "
        "of a company, you get send this text, you need to improve it: Everything we "
        "do at ArtConnect is grounded in an artists-first mindset, cultivating a "
        "culture of innovation and a shared purpose to leave an enduring impact. Firt "
        "and foremost our mission is to support artist by connecting them to "
        "opportunities, organizations people in the arts who help them thrive. Besides "
        "that goal our vision it to bring the arts together across the world and build "
        "a platform that can fortest collaboaration across the world with artist and "
        "art professional across all mediums and stages",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": '"In recent years, there have been significant advancements in '
        'renewable energy technology. Can you explain how solar panels work?"',
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "A DAO (Decentralized Autonomous Organization) can be turned into a "
        "credit buying debt by utilizing the principles of SVP/SVT",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "rewrite this problem into a problem definition for esg reporting "
        "Individuals with reading or learning difficulties struggle to comprehend "
        "written material, hindering their ability to learn and succeed in academic "
        "and professional settings.",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "can you write a blog post about how AI image generators can help "
        "video productions create mood boards, and include a quick summary of what "
        "mood boards are and their benefits",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "If the function descriptions of a web application are as follows. "
        "Could you recommend black box testing cases to identify the application's "
        "security vulnerabilities? A person without an account can register as a new "
        "user. After registering, the user must confirm their email before they can "
        "log in and start using the application. The following attributes are required "
        "when registering: - Email - Username - Password - Account type (refugee or "
        "volunteer)",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "About the frustrated reflection, i understand that i need three "
        "adjacent mediums. I will have two interfaces, if I send a plane wave from the "
        "first medium to the first interface with an angle such that it gets reflected "
        "totally,i'll create an evanescent wave in the second medium. If the thickness "
        "of the second medium is so low that the evanescent wave reaches the second "
        "interface with a non zero amplitude, then imposing the boundary conditions "
        "will result in an homogeneous wave in the third medium, if and only if the "
        "parallel component of the wave vector in the second medium is lower than the "
        "wave vector value in the third medium. Is that correct? Could you explain it "
        "again if you find any errors?",
    },
    {
        "system": "You are a helpful assistant. Provide a thorough, detailed response.",
        "user": "Let's pretend you are teaching 2nd year undergraduate engineers who "
        "are working on simple autonomous robots with wheels. The robots are equipped "
        "with an Arduino, 2 wheel encoders, an ultrasonic sensor and 2 infra red "
        "sensors and an IMU. Could you give some ideas for challenges to test the "
        "students robots?",
    },
]
