import json
import re
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

def get_chatbot_response(message, stage, risk):
    message = message.lower()
    stage = int(stage) if str(stage).isdigit() else 0
    
    # Keyword matching rules
    if re.search(r'\b(diet|food|eat|nutrition)\b', message):
        if stage >= 3:
            return "At an advanced stage, a strict low-sodium diet (<2g/day) is highly recommended to manage fluid retention (ascites). Ensure adequate lean protein intake to prevent muscle wasting, but consult a dietitian to avoid taxing the liver."
        else:
            return "For early-stage liver disease, a balanced Mediterranean-style diet rich in antioxidants, lean proteins, and low saturated fats is optimal. Avoid processed foods and added sugars."
            
    elif re.search(r'\b(alcohol|drink|beer|wine|liquor)\b', message):
        return "Absolute alcohol abstinence is critical for any stage of liver cirrhosis. Alcohol directly accelerates liver scarring and can trigger life-threatening decompensation."

    elif re.search(r'\b(cure|reverse|heal)\b', message):
        if stage >= 3:
            return "Advanced cirrhosis (Stage 3-4) is generally irreversible. The goal of treatment is to manage symptoms, slow progression, and evaluate for a liver transplant."
        else:
            return "Early-stage fibrosis (Stage 1-2) can sometimes be halted or even partially reversed by eliminating the underlying cause (e.g., stopping alcohol, treating hepatitis) and maintaining a healthy lifestyle."

    elif re.search(r'\b(exercise|workout|activity)\b', message):
        if stage >= 3:
            return "Gentle, low-impact activity like short walks is okay, but avoid heavy lifting or high-impact sports, especially if you have low platelets or varices (risk of bleeding)."
        else:
            return "Regular, moderate exercise is highly beneficial. It helps manage weight, reduces liver fat, and improves overall cardiovascular health."

    elif re.search(r'\b(transplant|surgery)\b', message):
        if stage >= 3:
            return "Because you are at a higher risk stage, regular monitoring by a hepatologist is vital. If your liver begins to fail (decompensation), you may undergo an evaluation to join the liver transplant waiting list."
        else:
            return "Liver transplants are typically reserved for patients with end-stage liver disease or acute liver failure. At your current early stage, focus on medical management and lifestyle changes to prevent progression."

    elif re.search(r'\b(medication|medicine|drugs|pills)\b', message):
        return "Never take over-the-counter medications (especially NSAIDs like ibuprofen or high doses of Tylenol) without consulting your doctor. The liver processes most drugs, and a damaged liver can lead to dangerous drug toxicity."
        
    elif re.search(r'\b(symptom|feel|pain)\b', message):
        if stage >= 3:
            return "Advanced stage symptoms can include jaundice (yellowing), severe fatigue, abdominal swelling (ascites), confusion (encephalopathy), and easy bruising. Seek emergency care for vomiting blood or severe confusion."
        else:
            return "Early stages often have no symptoms, or non-specific symptoms like mild fatigue or right upper quadrant discomfort."
            
    elif re.search(r'\b(stage|risk|meaning)\b', message):
        if stage == 1:
            return "Stage 1 indicates early fibrosis where the liver has some scarring but is still functioning relatively well (Low Risk)."
        elif stage == 2:
            return "Stage 2 indicates moderate fibrosis. Scarring is more pronounced, but the liver usually still compensates (Medium Risk)."
        elif stage == 3:
            return "Stage 3 indicates advanced fibrosis or early cirrhosis. The liver is heavily scarred and function is becoming impaired (High Risk)."
        elif stage == 4:
            return "Stage 4 is critical cirrhosis, often decompensated. The liver is failing to perform its crucial functions, leading to severe complications."
        else:
            return f"Cirrhosis progression is classified into 4 stages: Stage 1 (Mild) to Stage 4 (Decompensated/Critical). Your current assessment status is {risk} Risk."


    else:
        return f"I understand your concern. Based on your predicted Stage {stage} ({risk} Risk), it's important to discuss this specific question directly with your hepatologist or primary care physician, as individual cases vary greatly."

@csrf_exempt
def ask_chatbot(request):
    """
    API Endpoint for the interactive React/JS Chatbot on the Frontend.
    Expects JSON: { "message": "user question", "stage": "3", "risk": "High" }
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            stage = data.get('stage', '0')
            risk = data.get('risk', 'Unknown')
            
            if not user_message:
                return JsonResponse({'error': 'Message cannot be empty'}, status=400)
                
            bot_reply = get_chatbot_response(user_message, stage, risk)
            
            return JsonResponse({'reply': bot_reply})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)
