from typing import List, Optional
from pydantic import BaseModel, Field

class Coding(BaseModel):
    system: Optional[str] = None
    code: Optional[str] = None
    display: Optional[str] = None

class CodeableConcept(BaseModel):
    coding: List[Coding] = []
    text: Optional[str] = None

class Reference(BaseModel):
    reference: str
    type: Optional[str] = None
    display: Optional[str] = None

class Quantity(BaseModel):
    value: float
    unit: Optional[str] = None
    system: Optional[str] = None
    code: Optional[str] = None

# FHIR v4 Observation resource
class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: Optional[str] = None
    status: str = "final"
    category: List[CodeableConcept] = []
    code: CodeableConcept
    subject: Reference
    effectiveDateTime: str
    valueQuantity: Optional[Quantity] = None
    component: Optional[List[dict]] = None

def create_fhir_steps_observation(patient_id: str, steps: int, date_str: str) -> FHIRObservation:
    """
    Creates a FHIR v4 Observation resource for step counts.
    LOINC Code: 55423-8 (Number of steps in 24 hour period)
    """
    return FHIRObservation(
        status="final",
        category=[
            CodeableConcept(
                coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="physical-activity", display="Physical Activity")]
            )
        ],
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="55423-8", display="Number of steps in 24 hour period")]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        effectiveDateTime=date_str,
        valueQuantity=Quantity(value=float(steps), unit="steps", system="http://unitsofmeasure.org", code="steps")
    )

def create_fhir_sleep_observation(patient_id: str, hours: float, date_str: str) -> FHIRObservation:
    """
    Creates a FHIR v4 Observation resource for sleep duration.
    LOINC Code: 24826-6 (Hours of sleep)
    """
    return FHIRObservation(
        status="final",
        category=[
            CodeableConcept(
                coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="vital-signs", display="Vital Signs")]
            )
        ],
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="24826-6", display="Hours of sleep")]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        effectiveDateTime=date_str,
        valueQuantity=Quantity(value=hours, unit="h", system="http://unitsofmeasure.org", code="h")
    )

def create_fhir_nutrition_observation(patient_id: str, calories: float, protein_g: float, carbs_g: float, fat_g: float, date_str: str) -> FHIRObservation:
    """
    Creates a FHIR v4 Observation resource for daily nutritional logs.
    LOINC Code: 9052-2 (Calorie intake 24h) with components for macronutrients.
    """
    return FHIRObservation(
        status="final",
        category=[
            CodeableConcept(
                coding=[Coding(system="http://terminology.hl7.org/CodeSystem/observation-category", code="lifestyle", display="Lifestyle")]
            )
        ],
        code=CodeableConcept(
            coding=[Coding(system="http://loinc.org", code="9052-2", display="Calorie intake 24 hour")]
        ),
        subject=Reference(reference=f"Patient/{patient_id}"),
        effectiveDateTime=date_str,
        valueQuantity=Quantity(value=calories, unit="kcal", system="http://unitsofmeasure.org", code="kcal"),
        component=[
            {
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "9061-3", "display": "Protein intake 24 hour"}]
                },
                "valueQuantity": {"value": protein_g, "unit": "g", "system": "http://unitsofmeasure.org", "code": "g"}
            },
            {
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "9058-9", "display": "Carbohydrate intake 24 hour"}]
                },
                "valueQuantity": {"value": carbs_g, "unit": "g", "system": "http://unitsofmeasure.org", "code": "g"}
            },
            {
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "9060-5", "display": "Fat intake 24 hour"}]
                },
                "valueQuantity": {"value": fat_g, "unit": "g", "system": "http://unitsofmeasure.org", "code": "g"}
            }
        ]
    )
