import { SignUp as ClerkSignUp } from "@clerk/clerk-react";
import { Link } from "react-router-dom";
import { Music, Sparkles, ArrowLeft } from "lucide-react";

const SignUp = () => {
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
            Create your account
          </h1>
          <p className="text-muted-foreground">
            Start transforming music into chords with AI
          </p>
        </div>

        {/* Clerk Sign Up Component */}
        <div className="animate-fade-in-scale opacity-0 stagger-2">
          <ClerkSignUp 
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
                form: "clerk-form space-y-4",
                formFieldRow: "clerk-field-row",
                formFieldRow__name: "grid grid-cols-2 gap-3",
                formField__firstName: "flex-1",
                formField__lastName: "flex-1",
                formFieldLabel: "text-sm font-medium text-gray-300 mb-1.5 block",
                formFieldLabelRow: "flex items-center justify-between mb-1.5",
                formFieldHintText: "text-xs text-gray-500",
                formFieldInput: "clerk-input",
                formFieldInputShowPasswordButton: "text-gray-400 hover:text-white transition-colors",
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
            signInUrl="/login"
          />
        </div>

        {/* Sign In Link */}
        <div className="mt-6 text-center animate-fade-in-scale opacity-0 stagger-3">
          <p className="text-muted-foreground">
            Already have an account?{" "}
            <Link to="/login" className="text-primary hover:text-accent font-semibold transition-colors">
              Sign in
            </Link>
          </p>
        </div>

        {/* Terms */}
        <p className="text-center text-muted-foreground text-xs mt-6 animate-fade-in-scale opacity-0 stagger-4 px-4">
          By creating an account, you agree to our{" "}
          <a href="#" className="text-primary hover:text-accent transition-colors underline underline-offset-2">
            Terms of Service
          </a>
          {" "}and{" "}
          <a href="#" className="text-primary hover:text-accent transition-colors underline underline-offset-2">
            Privacy Policy
          </a>
        </p>
      </div>
    </div>
  );
};

export default SignUp;
