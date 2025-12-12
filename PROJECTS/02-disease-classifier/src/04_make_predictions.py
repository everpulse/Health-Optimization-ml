"""
Prediction Module - PROFESSIONAL VERSION
Make predictions on new patient data using trained models
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import sys
import re


# Persian to English number mapping
PERSIAN_NUMBERS = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
}

# Arabic to English number mapping
ARABIC_NUMBERS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}


def persian_to_english(text):
    """Convert Persian/Arabic numbers to English"""
    text = str(text)
    
    # Convert Persian numbers
    for persian, english in PERSIAN_NUMBERS.items():
        text = text.replace(persian, english)
    
    # Convert Arabic numbers
    for arabic, english in ARABIC_NUMBERS.items():
        text = text.replace(arabic, english)
    
    return text


def validate_number(value, min_val, max_val, name):
    """
    Validate a number is within range
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the field (for error messages)
    
    Returns:
        tuple: (is_valid, converted_value, error_message)
    """
    # Convert Persian/Arabic numbers
    value = persian_to_english(value)
    
    try:
        # Try to convert to float
        num = float(value)
        
        # Check range
        if num < min_val or num > max_val:
            return False, None, f"❌ {name} must be between {min_val} and {max_val}"
        
        return True, num, None
        
    except ValueError:
        return False, None, f"❌ Invalid input! {name} must be a number"


def get_input_with_retry(prompt, min_val, max_val, name, is_int=False):
    """
    Get user input with validation and retry
    
    Args:
        prompt: Input prompt message
        min_val: Minimum valid value
        max_val: Maximum valid value
        name: Field name for error messages
        is_int: Whether to return integer
    
    Returns:
        Validated number
    """
    while True:
        try:
            user_input = input(prompt).strip()
            
            # Allow user to quit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n❌ Input cancelled by user")
                sys.exit(0)
            
            # Validate
            is_valid, value, error_msg = validate_number(user_input, min_val, max_val, name)
            
            if is_valid:
                if is_int:
                    return int(value)
                return value
            else:
                print(error_msg)
                print(f"📌 Try again (or type 'quit' to exit)\n")
                
        except KeyboardInterrupt:
            print("\n\n❌ Input cancelled by user")
            sys.exit(0)


def load_model(model_path):
    """Load a trained model"""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model not found - {model_path}")
        print("   Please run 02-train_models.py first!")
        sys.exit(1)


def load_scaler(models_dir):
    """Load the saved scaler"""
    scaler_path = models_dir / 'scaler.pkl'
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except FileNotFoundError:
        print(f"❌ Error: Scaler not found - {scaler_path}")
        print("   Please run 02-train_models.py first!")
        sys.exit(1)


def get_patient_input():
    """
    Get patient data from user input with validation
    
    Returns:
        dict: Patient data
    """
    print("\n" + "="*70)
    print("📋 ENTER PATIENT DATA")
    print("="*70)
    print("💢 Type 'quit' at any time to exit")
    print("-" * 70)
    
    try:
        pregnancies = get_input_with_retry(
            "Pregnancies (0-17): ",
            min_val=0, max_val=17,
            name="Pregnancies",
            is_int=True
        )
        
        glucose = get_input_with_retry(
            "Glucose level (0-200 mg/dL): ",
            min_val=0, max_val=200,
            name="Glucose"
        )
        
        blood_pressure = get_input_with_retry(
            "Blood Pressure (0-122 mm Hg): ",
            min_val=0, max_val=122,
            name="Blood Pressure"
        )
        
        skin_thickness = get_input_with_retry(
            "Skin Thickness (0-99 mm): ",
            min_val=0, max_val=99,
            name="Skin Thickness"
        )
        
        insulin = get_input_with_retry(
            "Insulin (0-846 μU/ml): ",
            min_val=0, max_val=846,
            name="Insulin"
        )
        
        bmi = get_input_with_retry(
            "BMI (0-67 kg/m²): ",
            min_val=0, max_val=67,
            name="BMI"
        )
        
        dpf = get_input_with_retry(
            "Diabetes Pedigree Function (0.0-2.5): ",
            min_val=0.0, max_val=2.5,
            name="Diabetes Pedigree Function"
        )
        
        age = get_input_with_retry(
            "Age (21-81 years): ",
            min_val=21, max_val=81,
            name="Age",
            is_int=True
        )
        
        patient_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        return patient_data
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None


def create_sample_patients():
    """Create sample patient data for testing"""
    samples = [
        {
            'Name': '🟩 Healthy Patient (Low Risk)',
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 66,
            'SkinThickness': 29,
            'Insulin': 85,
            'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351,
            'Age': 31
        },
        {
            'Name': '⚠️ High Risk Patient',
            'Pregnancies': 10,
            'Glucose': 168,
            'BloodPressure': 74,
            'SkinThickness': 46,
            'Insulin': 230,
            'BMI': 38.0,
            'DiabetesPedigreeFunction': 1.191,
            'Age': 56
        },
        {
            'Name': '🟨 Moderate Risk Patient',
            'Pregnancies': 4,
            'Glucose': 110,
            'BloodPressure': 76,
            'SkinThickness': 20,
            'Insulin': 100,
            'BMI': 28.4,
            'DiabetesPedigreeFunction': 0.118,
            'Age': 27
        },
        {
            'Name': '🟥 Very High Risk Patient',
            'Pregnancies': 8,
            'Glucose': 196,
            'BloodPressure': 76,
            'SkinThickness': 29,
            'Insulin': 280,
            'BMI': 37.5,
            'DiabetesPedigreeFunction': 0.605,
            'Age': 57
        }
    ]
    
    return samples


def feature_engineering(df):
    """Apply same feature engineering as training"""
    # Same 8 features as training
    df['BMI_Age_Interaction'] = (df['BMI'] * df['Age']) / 100
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1)
    df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 1)
    
    df['BMI_Squared'] = df['BMI'] ** 2
    df['Age_Squared'] = df['Age'] ** 2
    
    df['Age_Risk'] = pd.cut(df['Age'], bins=[0, 30, 45, 100], labels=[0, 1, 2])
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 25, 30, 100], labels=[0, 1, 2])
    df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 100, 126, 200], labels=[0, 1, 2])
    
    return df


def prepare_input(patient_data, scaler):
    """
    Prepare patient data for prediction with feature engineering and scaling
    
    Args:
        patient_data: Dictionary with patient features
        scaler: Trained StandardScaler
        
    Returns:
        DataFrame: Scaled features ready for prediction
    """
    # Feature columns in correct order
    feature_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Create DataFrame
    df = pd.DataFrame([patient_data], columns=feature_columns)
    
    # Apply feature engineering (same as training)
    df = feature_engineering(df)
    
    # Scale features
    X_scaled = scaler.transform(df)
    X_scaled = pd.DataFrame(X_scaled, columns=df.columns)
    
    return X_scaled


def make_prediction(model, X, model_name):
    """
    Make prediction for a patient
    
    Args:
        model: Trained model
        X: Prepared and scaled features
        model_name: Name of the model
        
    Returns:
        dict: Prediction results
    """
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        prob_healthy = probabilities[0] * 100
        prob_diabetes = probabilities[1] * 100
    else:
        prob_healthy = None
        prob_diabetes = None
    
    result = {
        'model': model_name,
        'prediction': 'Diabetes' if prediction == 1 else 'Healthy',
        'prediction_code': int(prediction),
        'prob_healthy': prob_healthy,
        'prob_diabetes': prob_diabetes
    }
    
    return result


def print_patient_info(patient_data):
    """Print patient information in a nice format"""
    print("\n" + "="*70)
    print("👤 PATIENT INFORMATION")
    print("="*70)
    
    if 'Name' in patient_data:
        print(f"\n{patient_data['Name']}")
        print("-" * 70)
    
    print(f"{'Pregnancies:':<35} {patient_data['Pregnancies']}")
    print(f"{'Glucose:':<35} {patient_data['Glucose']} mg/dL")
    print(f"{'Blood Pressure:':<35} {patient_data['BloodPressure']} mm Hg")
    print(f"{'Skin Thickness:':<35} {patient_data['SkinThickness']} mm")
    print(f"{'Insulin:':<35} {patient_data['Insulin']} μU/ml")
    print(f"{'BMI:':<35} {patient_data['BMI']} kg/m²")
    print(f"{'Diabetes Pedigree Function:':<35} {patient_data['DiabetesPedigreeFunction']}")
    print(f"{'Age:':<35} {patient_data['Age']} years")


def get_risk_level(prob_diabetes):
    """Determine risk level based on probability"""
    if prob_diabetes is None:
        return "Unknown", "⚪"
    elif prob_diabetes < 30:
        return "Low", "🟢"
    elif prob_diabetes < 50:
        return "Moderate", "🟡"
    elif prob_diabetes < 70:
        return "High", "🟠"
    else:
        return "Very High", "🔴"


def print_predictions(results):
    """Print prediction results from all models"""
    print("\n" + "="*70)
    print("🔮 PREDICTIONS FROM ALL MODELS")
    print("="*70)
    
    for result in results:
        print(f"\n{'─' * 70}")
        print(f"📊 {result['model']}")
        print(f"{'─' * 70}")
        
        # Prediction
        if result['prediction'] == 'Diabetes':
            print(f"Prediction: 🔴 {result['prediction']}")
        else:
            print(f"Prediction: 🟢 {result['prediction']}")
        
        # Probabilities and risk level (if available)
        if result['prob_healthy'] is not None:
            risk_level, emoji = get_risk_level(result['prob_diabetes'])
            
            print(f"\nConfidence Scores:")
            print(f"  Healthy:   {result['prob_healthy']:>6.2f}% {'█' * int(result['prob_healthy']/5)}")
            print(f"  Diabetes:  {result['prob_diabetes']:>6.2f}% {'█' * int(result['prob_diabetes']/5)}")
            print(f"\nRisk Level: {emoji} {risk_level}")


def get_consensus(results):
    """Get consensus prediction from all models"""
    predictions = [r['prediction_code'] for r in results]
    
    diabetes_votes = sum(predictions)
    healthy_votes = len(predictions) - diabetes_votes
    
    consensus = 'Diabetes' if diabetes_votes > healthy_votes else 'Healthy'
    confidence = max(diabetes_votes, healthy_votes) / len(predictions) * 100
    
    # Calculate average probability
    probs = [r['prob_diabetes'] for r in results if r['prob_diabetes'] is not None]
    avg_prob = np.mean(probs) if probs else None
    
    return {
        'consensus': consensus,
        'confidence': confidence,
        'avg_probability': avg_prob,
        'diabetes_votes': diabetes_votes,
        'healthy_votes': healthy_votes,
        'total_models': len(predictions)
    }


def print_consensus(consensus):
    """Print consensus results with visual bars"""
    print("\n" + "="*70)
    print("🎯 ENSEMBLE CONSENSUS")
    print("="*70)
    
    print(f"\n{'Voting Results:':<20}")
    
    total = consensus['total_models']
    healthy_bar = '█' * consensus['healthy_votes'] + '░' * (total - consensus['healthy_votes'])
    diabetes_bar = '█' * consensus['diabetes_votes'] + '░' * (total - consensus['diabetes_votes'])
    
    print(f"  🟢 Healthy:  {consensus['healthy_votes']}/{total}  [{healthy_bar}]")
    print(f"  🔴 Diabetes: {consensus['diabetes_votes']}/{total}  [{diabetes_bar}]")
    
    if consensus['avg_probability'] is not None:
        risk_level, emoji = get_risk_level(consensus['avg_probability'])
        print(f"\n{'Average Risk Probability:':<30} {consensus['avg_probability']:.1f}%")
        print(f"{'Risk Level:':<30} {emoji} {risk_level}")
    
    print(f"\n{'='*70}")
    if consensus['consensus'] == 'Diabetes':
        print(f"⚠️  FINAL DIAGNOSIS: {consensus['consensus'].upper()}")
        print(f"⚠️  RECOMMENDATION: Consult a healthcare professional")
    else:
        print(f"✅ FINAL DIAGNOSIS: {consensus['consensus'].upper()}")
        print(f"✅ RECOMMENDATION: Continue healthy lifestyle")
    print(f"{'='*70}")
    print(f"Consensus Confidence: {consensus['confidence']:.1f}%")
    print(f"{'='*70}")


def predict_samples(models_info, models_dir, scaler):
    """Predict on sample patients"""
    print("\n" + "="*70)
    print("⿻ TESTING WITH SAMPLE PATIENTS")
    print("="*70)
    
    # Load all models
    models = {}
    for model_file, model_name in models_info:
        model_path = models_dir / model_file
        if model_path.exists():
            models[model_name] = load_model(model_path)
        else:
            print(f"⚠️ Skipping {model_name} - model not found")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Get sample patients
    samples = create_sample_patients()
    
    # Predict for each sample
    for idx, sample in enumerate(samples, 1):
        print(f"\n{'#'*20}")
        print(f"SAMPLE PATIENT {idx}/{len(samples)}")
        print(f"{'#'*20}")
        
        print_patient_info(sample)
        
        # Prepare input
        X = prepare_input(sample, scaler)
        
        # Make predictions with all models
        results = []
        for model_name, model in models.items():
            result = make_prediction(model, X, model_name)
            results.append(result)
        
        # Print results
        print_predictions(results)
        
        # Get and print consensus
        consensus = get_consensus(results)
        print_consensus(consensus)
        
        # Pause between patients (except last one)
        if idx < len(samples):
            input("\n➡️  Press Enter to continue to next patient...")


def predict_custom(models_info, models_dir, scaler):
    """Predict on custom patient input"""
    print("\n" + "="*70)
    print("🩺 CUSTOM PATIENT PREDICTION")
    print("="*70)
    
    # Load all models
    models = {}
    for model_file, model_name in models_info:
        model_path = models_dir / model_file
        if model_path.exists():
            models[model_name] = load_model(model_path)
        else:
            print(f"⚠️ Skipping {model_name} - model not found")
    
    if not models:
        print("❌ No models found!")
        return
    
    # Get patient input
    patient_data = get_patient_input()
    
    if patient_data is None:
        return
    
    # Print patient info
    print_patient_info(patient_data)
    
    # Prepare input
    X = prepare_input(patient_data, scaler)
    
    # Make predictions with all models
    results = []
    for model_name, model in models.items():
        result = make_prediction(model, X, model_name)
        results.append(result)
    
    # Print results
    print_predictions(results)
    
    # Get and print consensus
    consensus = get_consensus(results)
    print_consensus(consensus)
    
    # Ask if user wants to predict another
    print("\n" + "─"*70)
    another = input("Would you like to predict another patient? (yes/no): ").strip().lower()
    if another in ['yes', 'y', 'بله', 'آره']:
        predict_custom(models_info, models_dir, scaler)


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🔮 DIABETES PREDICTION SYSTEM")
    print("="*70)
    print("AI-powered disease prediction using ensemble machine learning")
    print("="*70)
    
    # Paths
    project_dir = Path(__file__).parent.parent
    models_dir = project_dir / 'models'
    
    # Load scaler
    print("\n↻ Loading models...")
    scaler = load_scaler(models_dir)
    print("✓ Scaler loaded")
    
    # Model information
    models_info = [
        ('logistic_regression.pkl', 'Logistic Regression'),
        ('decision_tree.pkl', 'Decision Tree'),
        ('random_forest.pkl', 'Random Forest'),
        ('knn.pkl', 'K-Nearest Neighbors'),
        ('gradient_boosting.pkl', 'Gradient Boosting'),
        ('svm.pkl', 'Support Vector Machine'),
        ('voting_classifier.pkl', 'Voting Classifier')
    ]
    
    # Count available models
    available_models = sum(1 for f, _ in models_info if (models_dir / f).exists())
    print(f"✓ {available_models}/{len(models_info)} models available")
    
    # Menu
    while True:
        print("\n" + "="*70)
        print("📋 CHOOSE PREDICTION MODE")
        print("="*70)
        print("1. Test with sample patients (4 examples)")
        print("2. Enter custom patient data")
        print("3. Exit")
        print("─"*70)
        
        try:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            
            # Support Persian numbers
            choice = persian_to_english(choice)
            
            if choice == '1':
                predict_samples(models_info, models_dir, scaler)
                
            elif choice == '2':
                predict_custom(models_info, models_dir, scaler)
                
            elif choice == '3':
                print("\nThank you for using the prediction system!ツ")
                print("Goodbye!👋\n")
                break
                
            else:
                print("\n❌ Invalid choice! Please enter 1, 2, or 3")
                continue
            
        except KeyboardInterrupt:
            print("\n\n❌ Program cancelled by user")
            print("👋 Goodbye!\n")
            break
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or contact support\n")


if __name__ == "__main__":
    main()