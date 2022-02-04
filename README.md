# Detecting Regulation Violations for an Indian Regulatory bodythrough multi label classification

The Securities and Exchange Board of India (SEBI) is the regulatory body for securities and commodities in India. SEBI creates, and enforces regulations that must be followed by all listed companies. To the best of our knowledge, this is the first work on identifying the regulation(s) that a SEBI-related case violates, which could be of substantial value to companies, lawyers, and other stakeholders in the regulatory process. We create a dataset for this task by automatically extracting violations from publicly available case-files. Using this data, we explore various multi-label text classification methods to determine the potentially multiple regulations violated by (the facts of) a case.Our experiments demonstrate the importance of employing contextual text representations to understand complex financial and legal concepts. We also highlight the challenges that must be addressed to develop a fully functional system in the real-world.

# Data 

- Unzip the `data.zip` file in the `data` folder
- Run `get_text_labels.py` in `src/` to get the text, label pairs 