import React, { useState } from 'react';
import './AnalysisInterface.css';

interface PersonalityScores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface PersonalityDescriptions {
  openness: string;
  conscientiousness: string;
  extraversion: string;
  agreeableness: string;
  neuroticism: string;
}

interface PersonalityResult {
  scores: PersonalityScores;
  descriptions: PersonalityDescriptions;
  analysis_duration: number;
  confidence: number;
}

interface DatingPreference {
  stimulus_type: string;
  attraction_score: number;
  attraction_level: string;
  components: {
    approach_motivation: number;
    attention_p300: number;
    arousal: number;
  };
}

interface AnalysisInterfaceProps {
  isConnected: boolean;
}

const DATING_STIMULI = [
  { type: 'blonde', label: 'Blonde Hair', image: '👱‍♀️' },
  { type: 'brunette', label: 'Brunette Hair', image: '👩‍🦳' },
  { type: 'redhead', label: 'Red Hair', image: '👩‍🦰' },
  { type: 'athletic', label: 'Athletic Build', image: '🏃‍♀️' },
  { type: 'curvy', label: 'Curvy Figure', image: '💃' },
  { type: 'tall', label: 'Tall Height', image: '🦒' },
  { type: 'petite', label: 'Petite Height', image: '🧚‍♀️' },
  { type: 'intellectual', label: 'Intellectual Look', image: '🤓' }
];

const AnalysisInterface: React.FC<AnalysisInterfaceProps> = ({ isConnected }) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState<string>('');
  const [personalityResult, setPersonalityResult] = useState<PersonalityResult | null>(null);
  const [datingResults, setDatingResults] = useState<DatingPreference[]>([]);
  const [currentStimulusIndex, setCurrentStimulusIndex] = useState(0);
  const [showingStimulus, setShowingStimulus] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);

  const sendResultsToScanner = async (personalityData: PersonalityResult, datingData: DatingPreference[]) => {
    try {
      // Send results to scanner frontend via booth backend
      const response = await fetch('http://localhost:5000/send-analysis-results', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'eeg_analysis_complete',
          personality: personalityData,
          dating_preferences: datingData,
          timestamp: new Date().toISOString()
        }),
      });

      if (response.ok) {
        console.log('✅ Results sent to scanner successfully');
      } else {
        console.error('❌ Failed to send results to scanner');
      }
    } catch (error) {
      console.error('Error sending results to scanner:', error);
    }
  };

  const startComprehensiveAnalysis = async () => {
    if (!isConnected) {
      alert('Please connect to EEG first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setPersonalityResult(null);
    setDatingResults([]);

    try {
      // Step 1: Personality Analysis
      setAnalysisStep('Analyzing personality traits...');
      
      // Show countdown
      for (let i = 3; i > 0; i--) {
        setAnalysisStep(`Starting analysis in ${i}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      setAnalysisStep('Recording baseline EEG activity...');
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Call personality analysis API
      const personalityResponse = await fetch('http://localhost:5000/analyze-personality', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const personalityData = await personalityResponse.json();
      
      if (personalityData.status === 'success') {
        setPersonalityResult(personalityData.personality);
        setAnalysisStep('Personality analysis complete! Starting preference testing...');
      } else {
        throw new Error(personalityData.error || 'Personality analysis failed');
      }

      await new Promise(resolve => setTimeout(resolve, 2000));

      // Step 2: Dating Preferences Analysis
      setAnalysisStep('Testing dating preferences...');
      const results: DatingPreference[] = [];

      for (let i = 0; i < DATING_STIMULI.length; i++) {
        const stimulus = DATING_STIMULI[i];
        setCurrentStimulusIndex(i);
        setAnalysisStep(`Testing preference ${i + 1} of ${DATING_STIMULI.length}...`);
        setShowingStimulus(true);

        // Show stimulus for 3 seconds
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        setShowingStimulus(false);

        // Analyze EEG response
        const response = await fetch('http://localhost:5000/analyze-dating-preference', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            stimulus_type: stimulus.type,
            duration: 3
          }),
        });

        const data = await response.json();
        
        if (data.status === 'success') {
          results.push(data.preference);
        }

        // Brief pause between stimuli
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      setDatingResults(results);
      setAnalysisStep('Analysis complete! Sending results...');
      
      // Step 3: Send results to scanner
      await sendResultsToScanner(personalityData.personality, results);
      
      setAnalysisComplete(true);
      setAnalysisStep('Results sent to scanner successfully!');
      
    } catch (error) {
      console.error('Comprehensive analysis error:', error);
      alert('Analysis failed. Please try again.');
      setAnalysisStep('Analysis failed');
    } finally {
      setIsAnalyzing(false);
      setShowingStimulus(false);
    }
  };

  const renderPersonalityResults = () => {
    if (!personalityResult) return null;

    return (
      <div className="analysis-results">
        <h3>🧠 Your Personality Profile</h3>
        <div className="personality-scores">
          {Object.entries(personalityResult.scores).map(([trait, score]) => (
            <div key={trait} className="personality-trait">
              <div className="trait-header">
                <span className="trait-name">{trait.charAt(0).toUpperCase() + trait.slice(1)}</span>
                <span className="trait-score">{score}%</span>
              </div>
              <div className="trait-bar">
                <div 
                  className="trait-fill"
                  style={{ width: `${score}%` }}
                />
              </div>
              <p className="trait-description">
                {personalityResult.descriptions[trait as keyof PersonalityDescriptions]}
              </p>
            </div>
          ))}
        </div>
        <div className="analysis-meta">
          <p>Confidence: {personalityResult.confidence}%</p>
          <p>Analysis Duration: {personalityResult.analysis_duration} minutes</p>
        </div>
      </div>
    );
  };

  const renderDatingResults = () => {
    if (datingResults.length === 0) return null;

    // Sort by attraction score
    const sortedResults = [...datingResults].sort((a, b) => b.attraction_score - a.attraction_score);

    return (
      <div className="analysis-results">
        <h3>💕 Your Dating Preferences</h3>
        <div className="dating-preferences">
          {sortedResults.map((result, index) => {
            const stimulus = DATING_STIMULI.find(s => s.type === result.stimulus_type);
            return (
              <div key={result.stimulus_type} className="preference-item">
                <div className="preference-header">
                  <span className="preference-emoji">{stimulus?.image}</span>
                  <span className="preference-label">{stimulus?.label}</span>
                  <span className="preference-score">{result.attraction_score}%</span>
                </div>
                <div className="preference-bar">
                  <div 
                    className="preference-fill"
                    style={{ 
                      width: `${result.attraction_score}%`,
                      backgroundColor: result.attraction_score >= 60 ? '#ff4757' : 
                                      result.attraction_score >= 40 ? '#ffa502' : '#747d8c'
                    }}
                  />
                </div>
                <p className="preference-level">{result.attraction_level}</p>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderStimulusDisplay = () => {
    if (!showingStimulus) return null;

    const stimulus = DATING_STIMULI[currentStimulusIndex];
    return (
      <div className="stimulus-overlay">
        <div className="stimulus-content">
          <div className="stimulus-emoji">{stimulus.image}</div>
          <h2>{stimulus.label}</h2>
          <p>Focus on this image...</p>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${((currentStimulusIndex + 1) / DATING_STIMULI.length) * 100}%` }}
            />
          </div>
          <p>{currentStimulusIndex + 1} of {DATING_STIMULI.length}</p>
        </div>
      </div>
    );
  };

  return (
    <div className="analysis-interface">
      <div className="analysis-header">
        <h2>🧠 Brain Analysis Suite</h2>
        <p>Discover your personality and dating preferences through EEG analysis</p>
      </div>

      <div className="analysis-controls">
        <button 
          className="analysis-btn comprehensive-btn"
          onClick={startComprehensiveAnalysis}
          disabled={!isConnected || isAnalyzing}
        >
          {isAnalyzing ? 'Analyzing...' : '🧠 Start EEG Analysis'}
        </button>
        
        {analysisComplete && (
          <div className="analysis-complete">
            <div className="success-icon">✅</div>
            <p>Analysis complete! Results sent to scanner.</p>
          </div>
        )}
      </div>

      {isAnalyzing && (
        <div className="analysis-progress">
          <div className="progress-spinner"></div>
          <p>{analysisStep}</p>
          {showingStimulus && (
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${((currentStimulusIndex + 1) / DATING_STIMULI.length) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {renderPersonalityResults()}
      {renderDatingResults()}
      {renderStimulusDisplay()}
    </div>
  );
};

export default AnalysisInterface;