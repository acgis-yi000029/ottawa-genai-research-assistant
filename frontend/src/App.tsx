import { GoogleOAuthProvider } from '@react-oauth/google';
import { createContext, useContext, useState } from 'react';
import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import './App.css';
import { GOOGLE_CLIENT_ID } from './config/googleAuth';
import { AuthProvider, useAuth } from './contexts/AuthContext';

import Navbar from './components/Navbar';
import ChatPage from './pages/ChatPage';
import DocumentUploadPage from './pages/DocumentUploadPage';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ReportPage from './pages/ReportPage';
import SettingsPage from './pages/SettingsPage';

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();
  
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};

interface LanguageContextType {
  language: 'en' | 'fr';
  setLanguage: (lang: 'en' | 'fr') => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

const translations = {
  en: {
    // Navigation
    'nav.home': 'Home',
    'nav.chat': 'Chat',
    'nav.upload': 'Upload',
    'nav.reports': 'Reports',
    'nav.settings': 'Settings',
    'nav.language': 'Language',
    'nav.login': 'Login',
    'nav.logout': 'Logout',
    'nav.profile': 'Profile',
    
    // Home page hero section
    'home.hero.badge': '🇨🇦 AI-Powered Research Assistant',
    'home.hero.title': 'Transform Your Research',
    'home.hero.titleHighlight': ' with AI',
    'home.hero.subtitle': 'Unlock the power of artificial intelligence to accelerate your research, analyze complex documents, and generate comprehensive insights in seconds.',
    'home.hero.startChat': 'Start Research Chat',
    'home.hero.uploadDocs': 'Upload Documents',
    'home.hero.demoTitle': 'AI Research Assistant',
    'home.hero.demoQuestion': 'What are the key findings about climate change in Ottawa?',
    'home.hero.demoResponse': 'Based on recent research, Ottawa faces significant climate impacts including increased temperatures and changing precipitation patterns...',
    
    // Features section
    'home.features.title': 'Powerful Research Capabilities',
    'home.features.subtitle': 'Everything you need to conduct advanced research with AI assistance',
    'home.features.qa.title': 'Intelligent Q&A',
    'home.features.qa.desc': 'Ask complex questions and get detailed, well-researched answers powered by advanced AI models.',
    'home.features.qa.benefit1': 'Natural language processing',
    'home.features.qa.benefit2': 'Context-aware responses',
    'home.features.analysis.title': 'Data Analysis',
    'home.features.analysis.desc': 'Upload documents and receive comprehensive analysis with key insights and summaries.',
    'home.features.analysis.benefit1': 'Multiple file formats',
    'home.features.analysis.benefit2': 'Visual data representation',
    'home.features.reports.title': 'Report Generation',
    'home.features.reports.desc': 'Generate professional research reports with citations, graphs, and detailed findings.',
    'home.features.reports.benefit1': 'Automated citations',
    'home.features.reports.benefit2': 'Export to multiple formats',
    
    // Why choose section
    'home.whyChoose.title': 'Why Researchers Choose Our Platform',
    'home.whyChoose.subtitle': 'Join thousands of researchers who have accelerated their work with our AI-powered tools.',
    'home.whyChoose.secure.title': 'Secure & Private',
    'home.whyChoose.secure.desc': 'Your research data is protected with enterprise-grade security and privacy measures.',
    'home.whyChoose.fast.title': 'Lightning Fast',
    'home.whyChoose.fast.desc': 'Get results in seconds, not hours. Our optimized AI models deliver rapid insights.',
    'home.whyChoose.bilingual.title': 'Bilingual Support',
    'home.whyChoose.bilingual.desc': 'Full support for English and French, perfect for Canadian research requirements.',
    'home.whyChoose.testimonial': '"This AI assistant has revolutionized my research process. What used to take days now takes hours."',
    'home.whyChoose.testimonialAuthor': 'Dr. Sarah Chen',
    'home.whyChoose.testimonialTitle': 'Research Scientist, University of Ottawa',
    
    // How it works section
    'home.howItWorks.title': 'How It Works',
    'home.howItWorks.subtitle': 'Get started with AI-powered research in three simple steps',
    'home.howItWorks.step1.title': 'Upload Your Documents',
    'home.howItWorks.step1.desc': 'Simply drag and drop your research documents, PDFs, or data files into our secure platform.',
    'home.howItWorks.step2.title': 'Ask Questions',
    'home.howItWorks.step2.desc': 'Use natural language to ask questions about your documents or request specific analysis.',
    'home.howItWorks.step3.title': 'Get Insights',
    'home.howItWorks.step3.desc': 'Receive comprehensive analysis, summaries, and visualizations based on your research needs.',
    
    // CTA section
    'home.cta.title': 'Ready to Accelerate Your Research?',
    'home.cta.subtitle': 'Join researchers worldwide who are using AI to unlock new insights and accelerate discovery.',
    'home.cta.startSession': 'Start Free Research Session',
    'home.cta.uploadDocs': 'Upload Documents',
    'home.cta.note': 'Trusted by 10,000+ researchers • No credit card required',

    // Legacy keys for backward compatibility
    'home.title': 'Ottawa Economic Development GenAI Research Assistant',
    'home.subtitle': 'Accelerate your economic research with intelligent AI-powered insights for Ottawa',
    'home.cta.start': 'Start Research Chat',
    'home.cta.upload': 'Upload Documents',
    
    // Features
    'features.qa.title': 'Intelligent Q&A',
    'features.qa.desc': 'Ask complex questions about Ottawa\'s economic landscape and get detailed, evidence-based answers',
    'features.analysis.title': 'Economic Analysis',
    'features.analysis.desc': 'Real-time analysis of economic data, trends, and market insights for Ottawa region',
    'features.reports.title': 'Smart Reports',
    'features.reports.desc': 'Automated generation of comprehensive research reports and data summaries',
    'features.accessibility.title': 'Accessible Design',
    'features.accessibility.desc': 'WCAG 2.1 compliant interface ensuring accessibility for all users',
    
    // Statistics
    'stats.documents': 'Documents',
    'stats.queries': 'Queries',
    'stats.languages': 'Languages',
    'stats.accessibility': 'Accessible',
    
    // Sections
    'sections.features': 'Key Features',
    'sections.features.desc': 'Powerful AI capabilities designed for economic development professionals',
    'sections.getstarted': 'Get Started',
    'sections.getstarted.desc': 'Choose your path to exploring economic development data',
    'sections.stats': 'Platform Statistics',
    
    // Home page sections
    'home.quickActions.title': 'Quick Actions',
    
    // Hero cards
    'hero.economic.title': 'Economic Trends',
    'hero.economic.desc': 'Real-time analysis',
    'hero.ai.title': 'AI Insights',
    'hero.ai.desc': 'Intelligent responses',
    'hero.reports.title': 'Smart Reports',
    'hero.reports.desc': 'Automated generation',
    
    // Welcome section
    'welcome.title': 'Welcome to Ottawa GenAI Research Assistant',
    'welcome.subtitle': 'Your intelligent research companion for Ottawa-related queries',
    'chat.placeholder': 'Ask me anything about Ottawa...',
    'upload.title': 'Upload Documents',
    'upload.description': 'Upload your research documents for analysis',
    'reports.title': 'Research Reports',
    'reports.description': 'View and manage your research reports',
    'settings.title': 'Settings',
    'settings.description': 'Configure your preferences',
    'login.title': 'Login',
    'login.subtitle': 'Sign in to access your research assistant',
    'login.google': 'Continue with Google',
    'login.email': 'Email',
    'login.password': 'Password',
    'login.submit': 'Sign In',
    'login.register': "Don't have an account? Sign up",
    'error.generic': 'An error occurred. Please try again.',
    'error.network': 'Network error. Please check your connection.',
    'error.auth': 'Authentication failed. Please check your credentials.',
    'success.login': 'Login successful!',
    'success.logout': 'Logout successful!',
    'loading': 'Loading...',
  },
  fr: {
    // Navigation
    'nav.home': 'Accueil',
    'nav.chat': 'Chat',
    'nav.upload': 'Télécharger',
    'nav.reports': 'Rapports',
    'nav.settings': 'Paramètres',
    'nav.language': 'Langue',
    'nav.login': 'Connexion',
    'nav.logout': 'Déconnexion',
    'nav.profile': 'Profil',
    
    // Home page hero section
    'home.hero.badge': '🇨🇦 Assistant de Recherche IA',
    'home.hero.title': 'Transformez Votre Recherche',
    'home.hero.titleHighlight': ' avec l\'IA',
    'home.hero.subtitle': 'Libérez la puissance de l\'intelligence artificielle pour accélérer votre recherche, analyser des documents complexes et générer des insights complets en quelques secondes.',
    'home.hero.startChat': 'Commencer le Chat de Recherche',
    'home.hero.uploadDocs': 'Télécharger des Documents',
    'home.hero.demoTitle': 'Assistant de Recherche IA',
    'home.hero.demoQuestion': 'Quelles sont les principales conclusions sur le changement climatique à Ottawa?',
    'home.hero.demoResponse': 'Basé sur des recherches récentes, Ottawa fait face à des impacts climatiques significatifs incluant l\'augmentation des températures et des changements de précipitations...',
    
    // Features section
    'home.features.title': 'Capacités de Recherche Puissantes',
    'home.features.subtitle': 'Tout ce dont vous avez besoin pour mener des recherches avancées avec l\'assistance IA',
    'home.features.qa.title': 'Q&R Intelligente',
    'home.features.qa.desc': 'Posez des questions complexes et obtenez des réponses détaillées et bien recherchées alimentées par des modèles IA avancés.',
    'home.features.qa.benefit1': 'Traitement du langage naturel',
    'home.features.qa.benefit2': 'Réponses sensibles au contexte',
    'home.features.analysis.title': 'Analyse de Données',
    'home.features.analysis.desc': 'Téléchargez des documents et recevez une analyse complète avec des insights et résumés clés.',
    'home.features.analysis.benefit1': 'Formats de fichiers multiples',
    'home.features.analysis.benefit2': 'Représentation visuelle des données',
    'home.features.reports.title': 'Génération de Rapports',
    'home.features.reports.desc': 'Générez des rapports de recherche professionnels avec citations, graphiques et résultats détaillés.',
    'home.features.reports.benefit1': 'Citations automatisées',
    'home.features.reports.benefit2': 'Export vers formats multiples',
    
    // Why choose section
    'home.whyChoose.title': 'Pourquoi les Chercheurs Choisissent Notre Plateforme',
    'home.whyChoose.subtitle': 'Rejoignez des milliers de chercheurs qui ont accéléré leur travail avec nos outils alimentés par l\'IA.',
    'home.whyChoose.secure.title': 'Sécurisé et Privé',
    'home.whyChoose.secure.desc': 'Vos données de recherche sont protégées avec des mesures de sécurité et de confidentialité de niveau entreprise.',
    'home.whyChoose.fast.title': 'Ultra Rapide',
    'home.whyChoose.fast.desc': 'Obtenez des résultats en quelques secondes, pas en heures. Nos modèles IA optimisés livrent des insights rapides.',
    'home.whyChoose.bilingual.title': 'Support Bilingue',
    'home.whyChoose.bilingual.desc': 'Support complet pour l\'anglais et le français, parfait pour les exigences de recherche canadiennes.',
    'home.whyChoose.testimonial': '"Cet assistant IA a révolutionné mon processus de recherche. Ce qui prenait des jours ne prend maintenant que des heures."',
    'home.whyChoose.testimonialAuthor': 'Dr. Sarah Chen',
    'home.whyChoose.testimonialTitle': 'Scientifique de Recherche, Université d\'Ottawa',
    
    // How it works section
    'home.howItWorks.title': 'Comment Ça Marche',
    'home.howItWorks.subtitle': 'Commencez avec la recherche alimentée par l\'IA en trois étapes simples',
    'home.howItWorks.step1.title': 'Téléchargez Vos Documents',
    'home.howItWorks.step1.desc': 'Glissez et déposez simplement vos documents de recherche, PDFs ou fichiers de données dans notre plateforme sécurisée.',
    'home.howItWorks.step2.title': 'Posez des Questions',
    'home.howItWorks.step2.desc': 'Utilisez le langage naturel pour poser des questions sur vos documents ou demander une analyse spécifique.',
    'home.howItWorks.step3.title': 'Obtenez des Insights',
    'home.howItWorks.step3.desc': 'Recevez une analyse complète, des résumés et des visualisations basés sur vos besoins de recherche.',
    
    // CTA section
    'home.cta.title': 'Prêt à Accélérer Votre Recherche?',
    'home.cta.subtitle': 'Rejoignez les chercheurs du monde entier qui utilisent l\'IA pour débloquer de nouveaux insights et accélérer la découverte.',
    'home.cta.startSession': 'Commencer une Session de Recherche Gratuite',
    'home.cta.uploadDocs': 'Télécharger des Documents',
    'home.cta.note': 'Fait confiance par plus de 10 000 chercheurs • Aucune carte de crédit requise',

    // Legacy keys for backward compatibility
    'home.title': 'Assistant de Recherche GenAI pour le Développement Économique d\'Ottawa',
    'home.subtitle': 'Accélérez vos recherches économiques avec des insights intelligents alimentés par l\'IA pour Ottawa',
    'home.cta.start': 'Commencer le Chat de Recherche',
    'home.cta.upload': 'Télécharger des Documents',
    
    // Features
    'features.qa.title': 'Q&R Intelligente',
    'features.qa.desc': 'Posez des questions complexes sur le paysage économique d\'Ottawa et obtenez des réponses détaillées et basées sur des preuves',
    'features.analysis.title': 'Analyse Économique',
    'features.analysis.desc': 'Analyse en temps réel des données économiques, tendances et insights de marché pour la région d\'Ottawa',
    'features.reports.title': 'Rapports Intelligents',
    'features.reports.desc': 'Génération automatisée de rapports de recherche complets et de résumés de données',
    'features.accessibility.title': 'Design Accessible',
    'features.accessibility.desc': 'Interface conforme WCAG 2.1 garantissant l\'accessibilité pour tous les utilisateurs',
    
    // Statistics
    'stats.documents': 'Documents',
    'stats.queries': 'Requêtes',
    'stats.languages': 'Langues',
    'stats.accessibility': 'Accessible',
    
    // Sections
    'sections.features': 'Fonctionnalités Clés',
    'sections.features.desc': 'Capacités IA puissantes conçues pour les professionnels du développement économique',
    'sections.getstarted': 'Commencer',
    'sections.getstarted.desc': 'Choisissez votre voie pour explorer les données de développement économique',
    'sections.stats': 'Statistiques de la Plateforme',
    
    // Home page sections
    'home.quickActions.title': 'Actions Rapides',
    
    // Hero cards
    'hero.economic.title': 'Tendances Économiques',
    'hero.economic.desc': 'Analyse en temps réel',
    'hero.ai.title': 'Insights IA',
    'hero.ai.desc': 'Réponses intelligentes',
    'hero.reports.title': 'Rapports Intelligents',
    'hero.reports.desc': 'Génération automatisée',
    
    // Welcome section
    'welcome.title': 'Bienvenue dans l\'Assistant de Recherche GenAI d\'Ottawa',
    'welcome.subtitle': 'Votre compagnon de recherche intelligent pour les questions liées à Ottawa',
    'chat.placeholder': 'Demandez-moi n\'importe quoi sur Ottawa...',
    'upload.title': 'Télécharger des Documents',
    'upload.description': 'Téléchargez vos documents de recherche pour analyse',
    'reports.title': 'Rapports de Recherche',
    'reports.description': 'Consultez et gérez vos rapports de recherche',
    'settings.title': 'Paramètres',
    'settings.description': 'Configurez vos préférences',
    'login.title': 'Connexion',
    'login.subtitle': 'Connectez-vous pour accéder à votre assistant de recherche',
    'login.google': 'Continuer avec Google',
    'login.email': 'Email',
    'login.password': 'Mot de passe',
    'login.submit': 'Se connecter',
    'login.register': 'Vous n\'avez pas de compte ? Inscrivez-vous',
    'error.generic': 'Une erreur s\'est produite. Veuillez réessayer.',
    'error.network': 'Erreur réseau. Veuillez vérifier votre connexion.',
    'error.auth': 'Échec de l\'authentification. Veuillez vérifier vos identifiants.',
    'success.login': 'Connexion réussie !',
    'success.logout': 'Déconnexion réussie !',
    'loading': 'Chargement...',
  }
};

function App() {
  const [language, setLanguage] = useState<'en' | 'fr'>('en');

  const t = (key: string): string => {
    return translations[language][key as keyof typeof translations[typeof language]] || key;
  };

  // Google OAuth Client ID validation and setup
  const googleClientId = GOOGLE_CLIENT_ID || 'placeholder-client-id';
  
  // Show warning if Client ID is not properly configured
  if (!GOOGLE_CLIENT_ID || GOOGLE_CLIENT_ID === 'your-google-client-id-here.apps.googleusercontent.com') {
    console.warn('⚠️ Google OAuth is not properly configured. Please set a valid REACT_APP_GOOGLE_CLIENT_ID in your .env file.');
  }

  return (
    <GoogleOAuthProvider clientId={googleClientId}>
      <AuthProvider>
        <LanguageContext.Provider value={{ language, setLanguage, t }}>
          <Router>
            <div className="App">
              <Navbar />
              <main className="main-content">
                <Routes>
                  <Route path="/login" element={<LoginPage />} />
                  <Route path="/register" element={<RegisterPage />} />
                  <Route 
                    path="/" 
                    element={<HomePage />}
                  />
                  <Route 
                    path="/chat" 
                    element={
                      <ProtectedRoute>
                        <ChatPage />
                      </ProtectedRoute>
                    } 
                  />
                  <Route 
                    path="/upload" 
                    element={
                      <ProtectedRoute>
                        <DocumentUploadPage />
                      </ProtectedRoute>
                    } 
                  />
                  <Route 
                    path="/reports" 
                    element={
                      <ProtectedRoute>
                        <ReportPage />
                      </ProtectedRoute>
                    } 
                  />
                  <Route 
                    path="/settings" 
                    element={
                      <ProtectedRoute>
                        <SettingsPage />
                      </ProtectedRoute>
                    } 
                  />
                </Routes>
              </main>
            </div>
          </Router>
        </LanguageContext.Provider>
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}

export default App;
