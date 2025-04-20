# Web view

## Expected file structure

Inside the `web` folder. From the repository's perspective: `./src/web`:
```
.
├── app.py
├── requirements.txt
├── resNet_model.keras
├── static
│   ├── css
│   │   ├── index_styles.css
│   │   └── result_styles.css
│   └── js
│       └── index_script.js
└── templates
    ├── index.html
    └── result.html

5 directories, 8 files
```

## Set up

Set up virtual environment:
`python -m venv venv`
`source venv/bin/activate`

Install dependencies:
`pip install -r requirements.txt`

Run Flask server:
`python app.py`

Open in browser:
`http://127.0.0.1:5000/`
