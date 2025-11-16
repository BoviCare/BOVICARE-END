import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';
import ProtectedRoute from '../components/ProtectedRoute';
import Login from "../components/auth/login/login";
import ForgotSelect from "../components/auth/forgot/ForgotSelect";
import ForgotCode from "../components/auth/forgot/ForgotCode";
import ForgotNewPassword from "../components/auth/forgot/ForgotNewPassword";
import Register from "../components/auth/register/register";
import Home from "../pages/Home/Home";
import Profile from '../pages/Profile/Profile';
import WeightTrackingPage from '../pages/WeightTrackingPage/WeightTrackingPage';
import CreateFarm from '../pages/CreateFarm/CreateFarm';
import Chat from '../pages/Chat/Chat';
import CadastrarGado from '../pages/CadastrarGado/CadastrarGado';
import Users from '../pages/Users/Users';
import ReportCattle from '../pages/ReportCattle/ReportCattle';

const RoutesApp = () => {
    return (
        <AuthProvider>
            <BrowserRouter>
                <Routes>
                    {/* Rotas públicas */}
                    <Route path="/login" element={<Login />} />
                    <Route path="/registro" element={<Register />} />
                    <Route path="/esqueci-senha" element={<ForgotSelect />} />
                    <Route path="/esqueci-senha/codigo" element={<ForgotCode />} />
                    <Route path="/esqueci-senha/nova-senha" element={<ForgotNewPassword />} />
                    
                    {/* Rotas protegidas */}
                    <Route path="/" element={
                        <ProtectedRoute>
                            <Home />
                        </ProtectedRoute>
                    } />
                    <Route path="/home" element={
                        <ProtectedRoute>
                            <Home />
                        </ProtectedRoute>
                    } />
                    <Route path="/profile" element={
                        <ProtectedRoute>
                            <Profile />
                        </ProtectedRoute>
                    } />
                    <Route path="/acompanhar-peso" element={
                        <ProtectedRoute>
                            <WeightTrackingPage />
                        </ProtectedRoute>
                    } />
                    <Route path="/criar-fazenda" element={
                        <ProtectedRoute>
                            <CreateFarm />
                        </ProtectedRoute>
                    } />
                    <Route path="/chat" element={
                        <ProtectedRoute>
                            <Chat />
                        </ProtectedRoute>
                    } />
                    <Route path="/cadastrar-gado" element={
                        <ProtectedRoute>
                            <CadastrarGado />
                        </ProtectedRoute>
                    } />
                    <Route path="/usuarios" element={
                        <ProtectedRoute>
                            <Users />
                        </ProtectedRoute>
                    } />
                    <Route path="/relatorio-gados" element={
                        <ProtectedRoute>
                            <ReportCattle />
                        </ProtectedRoute>
                    } />
                    
                    {/* Rota padrão - redirecionar para home se autenticado, senão para login */}
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
            </BrowserRouter>
        </AuthProvider>
    );
};

export default RoutesApp;
