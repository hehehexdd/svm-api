from fastapi import FastAPI
from app.api import router
from pydantic import ValidationError
from app.exception_handlers import validation_exception_handler, general_exception_handler

app = FastAPI()

app.include_router(router)

app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
