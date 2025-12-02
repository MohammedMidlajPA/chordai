import { ClerkProvider, SignedIn, SignedOut, RedirectToSignIn } from "@clerk/clerk-react";
import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// Lazy load pages for better performance
const Index = lazy(() => import("./pages/Index"));
const SignUp = lazy(() => import("./pages/SignUp"));
const Login = lazy(() => import("./pages/Login"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const NotFound = lazy(() => import("./pages/NotFound"));

const CLERK_PUBLISHABLE_KEY = "pk_test_YnJpZ2h0LW1hcnRpbi0zOS5jbGVyay5hY2NvdW50cy5kZXYk";
const queryClient = new QueryClient();

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => (
  <>
    <SignedIn>{children}</SignedIn>
    <SignedOut><RedirectToSignIn /></SignedOut>
  </>
);

const App = () => (
  <ClerkProvider publishableKey={CLERK_PUBLISHABLE_KEY}>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Suspense fallback={
            <div className="min-h-screen bg-background flex items-center justify-center">
              <div className="w-12 h-12 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
            </div>
          }>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/signup" element={<SignUp />} />
              <Route path="/login" element={<Login />} />
              <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ClerkProvider>
);

export default App;
