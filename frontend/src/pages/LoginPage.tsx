import { GoogleLogin } from '@react-oauth/google';
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './LoginPage.css';

const LoginPage: React.FC = () => {
  const { login, googleLogin, isLoading, error } = useAuth();
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [formError, setFormError] = useState<string | null>(null);
  const [showPassword, setShowPassword] = useState(false);

  // Add useEffect to dynamically style Google button
  useEffect(() => {
    const styleGoogleButton = () => {
      const googleButtons = document.querySelectorAll('.google-login-section [role="button"], .google-login-section button, .google-login-section div[tabindex="0"]');
      googleButtons.forEach((button: any) => {
        if (button) {
          button.style.width = '100%';
          button.style.height = '60px';
          button.style.borderRadius = '16px';
          button.style.border = '2px solid #e2e8f0';
          button.style.minHeight = '60px';
          button.style.maxHeight = '60px';
        }
      });
    };

    // Style immediately
    styleGoogleButton();
    
    // Also style after a short delay to handle dynamic loading
    const timer = setTimeout(styleGoogleButton, 500);
    
    return () => clearTimeout(timer);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    // Clear form error when user starts typing
    if (formError) {
      setFormError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);

    if (!formData.email || !formData.password) {
      setFormError('Please fill in all fields');
      return;
    }

    try {
      await login(formData);
      // 登录成功后跳转到主页
      navigate('/', { replace: true });
    } catch (error: any) {
      setFormError(error.message || 'Login failed');
    }
  };

  const handleGoogleSuccess = async (response: any) => {
    try {
      await googleLogin(response);
      // Google登录成功后跳转到主页
      navigate('/', { replace: true });
    } catch (error: any) {
      setFormError(error.message || 'Google login failed');
    }
  };

  const handleGoogleError = () => {
    setFormError('Google login failed. Please try again.');
  };

  return (
    <div className="login-page">
      {/* 背景装饰元素 */}
      <div className="background-decoration">
        <div className="floating-shape shape-1"></div>
        <div className="floating-shape shape-2"></div>
        <div className="floating-shape shape-3"></div>
        <div className="floating-shape shape-4"></div>
      </div>

      <div className="login-container">
        {/* Logo区域 */}
        <div className="login-logo">
          <div className="logo-icon">
            <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="60" height="60" rx="20" fill="url(#logoGradient)"/>
              <path d="M20 40V20H25V35H35V20H40V40H35V40C35 42.2091 33.2091 44 31 44H29C26.7909 44 25 42.2091 25 40H20Z" fill="white"/>
              <defs>
                <linearGradient id="logoGradient" x1="0" y1="0" x2="60" y2="60" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#004890"/>
                  <stop offset="1" stopColor="#0066cc"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
        </div>

        <div className="login-header">
          <h1>
            <span className="welcome-text">Welcome Back</span>
            <span className="subtitle-text">Sign in to access your research assistant</span>
          </h1>
        </div>

        <div className="login-form-container">
          {/* Google Login */}
          <div className="google-login-section">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={handleGoogleError}
              useOneTap={true}
              theme="outline"
              size="large"
              text="continue_with"
              shape="rectangular"
              width="100%"
            />
          </div>

          <div className="divider">
            <span>or</span>
          </div>

          {/* Email/Password Login */}
          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="email">
                <svg className="input-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M3 8L10.89 13.26C11.2187 13.4793 11.6049 13.5963 12 13.5963C12.3951 13.5963 12.7813 13.4793 13.11 13.26L21 8M5 19H19C19.5304 19 20.0391 18.7893 20.4142 18.4142C20.7893 18.0391 21 17.5304 21 17V7C21 6.46957 20.7893 5.96086 20.4142 5.58579C20.0391 5.21071 19.5304 5 19 5H5C4.46957 5 3.96086 5.21071 3.58579 5.58579C3.21071 5.96086 3 6.46957 3 7V17C3 17.5304 3.21071 18.0391 3.58579 18.4142C3.96086 18.7893 4.46957 19 5 19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                required
                disabled={isLoading}
                placeholder="Enter your email address"
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">
                <svg className="input-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M18 11H6C4.89543 11 4 11.8954 4 13V19C4 20.1046 4.89543 21 6 21H18C19.1046 21 20 20.1046 20 19V13C20 11.8954 19.1046 11 18 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M7 11V7C7 5.67392 7.52678 4.40215 8.46447 3.46447C9.40215 2.52678 10.6739 2 12 2C13.3261 2 14.5979 2.52678 15.5355 3.46447C16.4732 4.40215 17 5.67392 17 7V11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Password
              </label>
              <div className="password-input-container">
                <input
                  type={showPassword ? "text" : "password"}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  required
                  disabled={isLoading}
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M17.94 17.94C16.2306 19.243 14.1491 19.9649 12 20C5 20 1 12 1 12C2.24389 9.68192 4.028 7.66032 6.17 6.17M9.9 4.24C10.5883 4.0789 11.2931 3.99836 12 4C19 4 23 12 23 12C22.393 13.1356 21.6691 14.2048 20.84 15.19M14.12 14.12C13.8454 14.4148 13.5141 14.6512 13.1462 14.8151C12.7782 14.9791 12.3809 15.0673 11.9781 15.0744C11.5753 15.0815 11.1749 15.0074 10.8016 14.8565C10.4283 14.7056 10.0887 14.4811 9.80385 14.1962C9.51897 13.9113 9.29439 13.5717 9.14351 13.1984C8.99262 12.8251 8.91853 12.4247 8.92563 12.0219C8.93274 11.6191 9.02091 11.2218 9.18488 10.8538C9.34884 10.4858 9.58525 10.1546 9.88 9.88" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d="M1 1L23 23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {(error || formError) && (
              <div className="error-message">
                <svg className="error-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                  <line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" strokeWidth="2"/>
                  <line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" strokeWidth="2"/>
                </svg>
                {error || formError}
              </div>
            )}

            <button
              type="submit"
              className="login-button"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="button-spinner"></div>
                  Signing In...
                </>
              ) : (
                <>
                  <svg className="button-icon" width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M15 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H15M10 17L15 12L10 7M21 12H3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Sign In
                </>
              )}
            </button>
          </form>

          <div className="login-footer">
            <p>
              Don't have an account? 
              <Link to="/register" className="signup-link">
                <svg className="link-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M16 21V19C16 17.9391 15.5786 16.9217 14.8284 16.1716C14.0783 15.4214 13.0609 15 12 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21M12.5 7C12.5 9.20914 10.7091 11 8.5 11C6.29086 11 4.5 9.20914 4.5 7C4.5 4.79086 6.29086 3 8.5 3C10.7091 3 12.5 4.79086 12.5 7ZM20 8V14M23 11H17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Create Account
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
