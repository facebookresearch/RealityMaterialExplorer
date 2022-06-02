using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MasterChangeActiveObject : MonoBehaviour// : ChangeActiveObject
{
	
	private int countdown = 0;
	
	public ChangeActiveObject[] CAOs;
	public TranslateByRightStick[] TBRSs;
	public int activeIndex;
	private bool autofit;
	private int analyticBRDF;
	public ButtonGroupSelect ModelButtons;
	
	public void showInfoShader(){
		
		
	}
	
	public void setActiveIndex(int i){
		TBRSs[activeIndex].activeMovement = false;
		activeIndex=i;
		TBRSs[activeIndex].activeMovement = true;
		setAnalyticBRDF();
		updateMatCanvas();
	}
	
	public void setAnalyticBRDF(){
		analyticBRDF=CAOs[activeIndex].activeBRDFnumber;
		if(analyticBRDF<1){analyticBRDF=2;}
		if(analyticBRDF==2){ModelButtons.selectButton(0);}
		if(analyticBRDF==1){ModelButtons.selectButton(1);}
		if(analyticBRDF==3){ModelButtons.selectButton(2);}
	}

	public int getMERLnumber(){
		return CAOs[activeIndex].mi;
	}
	
	public string getMERLname(int i){
		return CAOs[activeIndex].getMERLname(i);
	}
	
	
	public void MERLnext(){
			CAOs[activeIndex].MERLnext();
	}
	
	public void MERLback(){ 
		CAOs[activeIndex].MERLback();
	}
	
	public void changeMERLMaterial(int i){
		CAOs[activeIndex].changeMERLMaterial(i);
	}
	
	//METALS
	public void metalnext(){
			CAOs[activeIndex].metalnext();
	}
	
	public void metalback(){ 
		CAOs[activeIndex].metalback();
	}
	
	

	public void changeBRDF(int BRDF){
		if(BRDF==-1){CAOs[activeIndex].changeBRDF(analyticBRDF);}
		else{CAOs[activeIndex].changeBRDF(BRDF);}
	}

	public void SetRoughness(float alpha){
		CAOs[activeIndex].SetRoughness(alpha);
	}
	
	public void SetAnisotropic(float alpha){
		CAOs[activeIndex].SetAnisotropic(alpha);
	}
	
	public void SetAnisotropicAxis(int axis){
		CAOs[activeIndex].SetAnisotropicAxis(axis);
	}
	
	public void SetSpecularScale(float s){
		CAOs[activeIndex].SetSpecularScale(s);
	}
	
	public void SetDiffuseRed(float dr){
		CAOs[activeIndex].SetDiffuseRed(dr);
	}
	
	public void SetDiffuseGreen(float dg){
		CAOs[activeIndex].SetDiffuseGreen(dg);
	}
	
	public void SetDiffuseBlue(float db){
		CAOs[activeIndex].SetDiffuseBlue(db);
	}
	
	public void SetSpecularRed(float sr){
		CAOs[activeIndex].SetSpecularRed(sr);
	}
	
	public void SetSpecularGreen(float sg){
		CAOs[activeIndex].SetSpecularGreen(sg);
	}
	
	public void SetSpecularBlue(float sb){
		CAOs[activeIndex].SetSpecularBlue(sb);
	}
	
	public void SetDiffuseScale(float d){
		CAOs[activeIndex].SetDiffuseScale(d);
	}
	
	public void SetFresnelF0(float f){
		CAOs[activeIndex].SetFresnelF0(f);
	}
	
	public void SetSMS(float t){
		CAOs[activeIndex].SetSMS(t);
	}
	
	public void SetMSVG(float t){
		CAOs[activeIndex].SetMSVG(t);
	}
	
	public void SetHeitz(float t){
		CAOs[activeIndex].SetHeitz(t);
	}
	
	public void SetGGXVG(float t){
		CAOs[activeIndex].SetGGXVG(t);
	}
	
	public void setActive(int index){
		CAOs[activeIndex].setActive(index);
	}
	
	public void setAutoFit(bool b){
		CAOs[activeIndex].setAutoFit(b);
	}
	
	public void maybeFit(){
		CAOs[activeIndex].maybeFit();
	}
	
	public void updateMatCanvas(){
		CAOs[activeIndex].updateMatCanvas();
	}
	
	
	public void pickFit(){
		CAOs[activeIndex].pickFit();
	}
	
	public void setFit(int model){
		CAOs[activeIndex].setFit(model);
	}
	
	public void setGGXfit(){
		CAOs[activeIndex].setGGXfit();
	}
	
	public void setCTfit(){
		CAOs[activeIndex].setCTfit();
	}
	
	public void setGGXVGfit(){
		CAOs[activeIndex].setGGXVGfit();
	}
	
	public void setGGXSMSfit(){
		CAOs[activeIndex].setGGXSMSfit();
	}
	
	public void setMSVGfit(){
		CAOs[activeIndex].setMSVGfit();
	}
	
	public void SetInterpMERL(float f){
		CAOs[activeIndex].SetInterpMERL(f);
	}
	
	public void SetFloatMERL(float f){
		CAOs[activeIndex].SetFloatMERL(f);
	}
	
	public void SaveSlotA(){
		CAOs[activeIndex].SaveSlotA();
	}
	
	public void SaveSlotB(){
		CAOs[activeIndex].SaveSlotB();
	}
	
	public void SaveSlotC(){
		CAOs[activeIndex].SaveSlotC();
	}
	
	public void LoadSlotA(){
		CAOs[activeIndex].LoadSlotA();
	}
	
	public void LoadSlotB(){
		CAOs[activeIndex].LoadSlotB();
	}
	
	public void LoadSlotC(){
		CAOs[activeIndex].LoadSlotC();
	}
	
	
	// Start is called before the first frame update
    void Start()
    {	
		activeIndex = 0;
		analyticBRDF=1;
    }
	

    // Update is called once per frame
    void Update()
    {
		if(countdown>0){
			countdown=countdown-1;
		}
		
		OVRInput.Update();
		if(OVRInput.Get(OVRInput.RawButton.B)){
			CAOs[activeIndex].SetShowInfoShader(1.0f);
		}
		else{
			CAOs[activeIndex].SetShowInfoShader(0.0f);
		}
    }
}
