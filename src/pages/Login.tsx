import { SignIn } from "@clerk/clerk-react";
import { Link } from "react-router-dom";
import { Music, Sparkles, ArrowLeft } from "lucide-react";

const Login = () => {
  return (
    <div className="min-h-screen hero-mesh flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-primary/15 blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-accent/15 blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-r from-primary/5 to-accent/5 blur-3xl" />
      </div>

      {/* Back to Home */}
      <Link 
        to="/" 
        className="absolute top-6 left-6 flex items-center space-x-2 text-muted-foreground hover:text-foreground transition-colors group z-20"
      >
        <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm font-medium">Back to home</span>
      </Link>

      <div className="w-full max-w-[420px] relative z-10">
        {/* Header */}
        <div className="text-center mb-8 animate-slide-up opacity-0">
          <Link to="/" className="inline-flex items-center space-x-3 mb-6 group">
            <div className="relative">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 border border-white/10 group-hover:scale-110 transition-all duration-300 shadow-lg shadow-primary/20">
                <Music className="h-7 w-7 text-primary" />
              </div>
              <Sparkles className="absolute -top-1 -right-1 h-4 w-4 text-accent animate-pulse" />
            </div>
            <span className="font-sora font-bold text-2xl bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              ChordAI
            </span>
          </Link>
          
          <h1 className="font-sora font-bold text-3xl sm:text-4xl mb-3 text-foreground">
            Welcome back
          </h1>
          <p className="text-muted-foreground">
            Sign in to continue your musical journey
          </p>
        </div>

        {/* Clerk Sign In Component */}
        <div className="animate-fade-in-scale opacity-0 stagger-2">
          <SignIn 
            appearance={{
              layout: {
                socialButtonsPlacement: "top",
                socialButtonsVariant: "blockButton",
              },
              elements: {
                rootBox: "w-full",
                card: "clerk-card",
                header: "hidden",
                main: "space-y-4",
                socialButtons: "space-y-3",
                socialButtonsBlockButton: "clerk-social-btn",
                socialButtonsBlockButtonText: "font-medium text-sm",
                socialButtonsProviderIcon: "w-5 h-5",
                dividerLine: "bg-white/10",
                dividerText: "text-muted-foreground text-xs uppercase tracking-wider bg-[#0d0d12] px-3",
                form: "space-y-4",
                formFieldRow: "space-y-4",
                formFieldLabel: "text-sm font-medium text-gray-300 mb-1.5 block",
                formFieldInput: "clerk-input",
                formFieldInputShowPasswordButton: "text-gray-400 hover:text-white transition-colors",
                formFieldAction: "text-primary hover:text-accent text-sm font-medium transition-colors",
                formButtonPrimary: "clerk-primary-btn",
                footerAction: "hidden",
                footer: "hidden",
                // Hide unwanted elements
                headerTitle: "hidden",
                headerSubtitle: "hidden",
                otpCodeField: "hidden",
                phoneInputBox: "hidden",
                alternativeMethods: "hidden",
                backLink: "hidden",
                identityPreview: "clerk-identity-preview",
                identityPreviewText: "text-white font-medium",
                identityPreviewEditButton: "text-primary hover:text-accent transition-colors",
              },
              variables: {
                colorPrimary: "#a78bfa",
                colorText: "#ffffff",
                colorTextSecondary: "#9ca3af",
                colorBackground: "#0d0d12",
                colorInputBackground: "rgba(255, 255, 255, 0.05)",
                colorInputText: "#ffffff",
                borderRadius: "0.75rem",
              }
            }}
            fallbackRedirectUrl="/dashboard"
            signUpUrl="/signup"
          />
        </div>

        {/* Sign Up Link */}
        <div className="mt-6 text-center animate-fade-in-scale opacity-0 stagger-3">
          <p className="text-muted-foreground">
            Don't have an account?{" "}
            <Link to="/signup" className="text-primary hover:text-accent font-semibold transition-colors">
              Sign up
            </Link>
          </p>
        </div>

        {/* Security Note */}
        <div className="mt-4 text-center animate-fade-in-scale opacity-0 stagger-4">
          <p className="text-muted-foreground text-xs flex items-center justify-center space-x-1.5">
            <svg className="w-3.5 h-3.5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span>Secured by Clerk</span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
