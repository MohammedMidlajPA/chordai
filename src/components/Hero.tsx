import { Link } from "react-router-dom";
import { Play, Upload, Zap, Sparkles, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-bg.jpg";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden hero-mesh">
      {/* Animated Background with Mesh */}
      <div className="absolute inset-0">
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            backgroundImage: `url(${heroImage})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        />
      </div>
      
      {/* Floating Orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-32 h-32 rounded-full bg-primary/20 blur-xl float animate-slide-up opacity-0 stagger-1" />
        <div className="absolute top-1/3 right-1/4 w-40 h-40 rounded-full bg-accent/15 blur-xl float-delayed animate-slide-up opacity-0 stagger-2" />
        <div className="absolute bottom-1/3 left-1/3 w-24 h-24 rounded-full bg-primary/25 blur-xl float-slow animate-slide-up opacity-0 stagger-3" />
        <div className="absolute bottom-1/4 right-1/3 w-36 h-36 rounded-full bg-accent/10 blur-xl float animate-slide-up opacity-0 stagger-4" />
      </div>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="max-w-6xl mx-auto">
          {/* Animated Badge */}
          <div className="inline-flex items-center space-x-2 sm:space-x-3 premium-card px-4 sm:px-6 py-2 sm:py-3 mb-6 sm:mb-8 text-xs sm:text-sm animate-fade-in-scale opacity-0 stagger-1 backdrop-blur-sm">
            <Sparkles className="h-4 w-4 sm:h-5 sm:w-5 text-primary pulse-glow" />
            <span className="text-foreground font-medium">AI-Powered Music Analysis</span>
            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-primary animate-pulse" />
          </div>

          {/* Hero Headlines */}
          <h1 className="font-sora font-bold text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl mb-6 sm:mb-8 text-foreground leading-[1.1] tracking-tight animate-slide-up opacity-0 stagger-2">
            Transform Any Song into{" "}
            <span className="relative inline-block">
              <span className="text-transparent bg-gradient-to-r from-primary via-accent to-primary bg-clip-text animate-pulse bg-300 bg-pos-0 hover:bg-pos-100 transition-all duration-2000">
                Chords
              </span>
              <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 to-accent/20 blur-xl -z-10 animate-pulse" />
            </span>{" "}
            <br className="hidden sm:block" />
            <span className="block sm:inline">Instantly</span>
          </h1>

          <p className="text-base sm:text-lg md:text-xl text-muted-foreground mb-8 sm:mb-12 max-w-2xl lg:max-w-3xl mx-auto leading-relaxed font-inter animate-slide-up opacity-0 stagger-3">
            Upload any track and get real-time, AI-powered chord analysis. 
            <span className="block mt-2 sm:inline sm:mt-0"> Perfect for musicians, producers, and music enthusiasts.</span>
          </p>

          {/* Enhanced CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 animate-slide-up opacity-0 stagger-4">
            <Link to="/signup" className="w-full sm:w-auto">
              <Button 
                size="lg" 
                className="w-full sm:w-auto magnetic-btn px-8 sm:px-12 py-4 sm:py-6 text-base sm:text-lg font-semibold rounded-xl sm:rounded-2xl group transition-all duration-300 hover:scale-105 hover:shadow-[0_0_30px_rgba(108,99,255,0.4)]"
              >
                <Upload className="h-5 w-5 sm:h-6 sm:w-6 mr-2 sm:mr-3 group-hover:scale-110 transition-transform duration-200" />
                Try It Free
                <ArrowRight className="h-4 w-4 sm:h-5 sm:w-5 ml-2 group-hover:translate-x-1 transition-transform duration-200" />
              </Button>
            </Link>
            <Button 
              variant="outline" 
              size="lg" 
              className="w-full sm:w-auto px-8 sm:px-12 py-4 sm:py-6 text-base sm:text-lg font-semibold premium-card border-primary/30 text-foreground hover:text-primary rounded-xl sm:rounded-2xl group transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(108,99,255,0.2)] hover:border-primary/60"
            >
              <Play className="h-5 w-5 sm:h-6 sm:w-6 mr-2 sm:mr-3 group-hover:scale-110 transition-transform duration-200" />
              Watch Demo
            </Button>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-primary/50 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-primary rounded-full mt-2 animate-pulse" />
        </div>
      </div>
    </section>
  );
};

export default Hero;
