The model predicts if person's salary is above or below 50K. The project is stored at GitHub, 
where GitHub Actions are used for CI. Namely flake8 is used to check for syntax error and pytest is used to run tests
before deployment. The FasAPI is used to expose functionality of the model. If CI finished successfully, then
it is automatically deployed to Heroku.

* DVC with S3 as remote repository.
* GitHub Actions for continious integration.
* FastAPI to create model API.
* Heroku to deploy API.
