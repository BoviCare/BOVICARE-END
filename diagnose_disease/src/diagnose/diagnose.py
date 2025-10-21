from diagnose.context import DiagnoseContext
from langchain_core.prompts import PromptTemplate
from diagnose.models import Diagnosis
import logging
from diagnose.get_syntoms import Symptoms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Diagnose:
    def __init__(self, ctx: DiagnoseContext):
        self.llm = ctx.llm
        self.schema = Diagnosis

    @property
    def prompt(self):
        template = """
            Você é um especialista em veterinária para gados. Sua tarefa é analisar sintomas de um animal e identificar
            quais os possíveis diagnósticos de doencas. Sugira em ordem quais são as que tem mais chances de ser a 
            doença do animal.
            Os sintomas do animal são: {symptoms}\n
            ----------------\n
        """

        return PromptTemplate(template=template)

    def diagnose(self):
        logger.info("Diagnosing...")
        symptoms = Symptoms().get_symptoms()
        logger.info(f"Symptoms: {symptoms}")
        chain = self.prompt | self.llm.with_structured_output(
            schema=self.schema, include_raw=False
        )
        try:
            diagnose = chain.invoke({"symptoms": symptoms})
            logger.info(f"Diagnosis: {diagnose}")
        except Exception as e:
            logger.error(f"Error diagnosing: {e}")
            raise e
        
    def run(self):
        self.diagnose()
